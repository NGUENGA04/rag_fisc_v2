"""Analyse de la question : intention, exercice(s), articles, montant.

Deux couches :
1. Extraction déterministe (expressions régulières) : années, articles, montants.
   Elle est fiable et gratuite, et sert de repli si le modèle échoue.
2. Mistral Small en sortie structurée : reformulation, intention, impôts visés.
Les deux résultats sont fusionnés.
"""

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from rag_fisc_v2.config import get_settings
from rag_fisc_v2.domain import LlmUsage
from rag_fisc_v2.ingestion.baremes import BAREME_KEYWORDS
from rag_fisc_v2.ingestion.references import extract_article_refs, extract_years, parse_article_id

log = logging.getLogger(__name__)

Intention = Literal["information", "calcul_bareme", "comparaison", "nouveautes", "hors_sujet"]
BaremeId = Literal["bareme-irpp", "bareme-tdl"]


class QueryAnalysis(BaseModel):
    question_reformulee: str = Field(
        description="Question reformulée de façon autonome et précise, avec les termes fiscaux exacts "
        "(ex. 'impôt sur les sociétés' plutôt que 'IS'), sans inventer d'information."
    )
    intention: Intention = Field(
        description="information : règle fiscale ; calcul_bareme : montant d'IRPP, CAC, CRTV, FNE, CCF ou TDL "
        "pour un salaire mensuel donné ; comparaison : évolution entre exercices ; nouveautes : "
        "changements apportés par une loi de finances ; hors_sujet : sans lien avec la fiscalité camerounaise."
    )
    exercices: list[int] = Field(default_factory=list, description="Exercices fiscaux mentionnés, ex. [2025].")
    articles: list[str] = Field(default_factory=list, description="Articles du CGI cités, ex. ['7', '92 bis'].")
    impots: list[str] = Field(default_factory=list, description="Impôts ou taxes concernés, ex. ['IS', 'TVA'].")
    bareme: BaremeId | None = Field(default=None, description="Barème à utiliser si intention = calcul_bareme.")
    montant: float | None = Field(default=None, description="Montant en FCFA mentionné (salaire mensuel, etc.).")


ANALYSIS_PROMPT = """Tu analyses des questions posées à un assistant fiscal spécialisé dans la fiscalité du Cameroun \
(Code Général des Impôts, lois de finances, circulaires de la DGI).
Remplis les champs demandés à partir de la question uniquement. N'invente aucune année, aucun article, aucun montant.
Sigles courants : IS (impôt sur les sociétés), IRPP (impôt sur le revenu des personnes physiques), TVA, \
AIR (acompte de l'impôt sur le revenu), CAC (centimes additionnels communaux), TDL (taxe de développement local), \
CRTV (redevance audiovisuelle), FNE (Fonds national de l'emploi), CCF (Crédit foncier), \
DSF (déclaration statistique et fiscale), \
BIC, BNC, RCM (revenus de capitaux mobiliers), DGI.
Un montant en millions doit être converti en FCFA (ex. '1,5 million' -> 1500000)."""


_AMOUNT = re.compile(
    r"(?P<num>\d{1,3}(?:[ . ]\d{3})+|\d{4,})(?:,(?P<dec>\d+))?\s*(?P<unit>millions?|M)?\s*(?:F\s?CFA|FCFA|francs?|F)?\b",
    re.IGNORECASE,
)
_MILLION = re.compile(r"(?P<num>\d+(?:[.,]\d+)?)\s*millions?", re.IGNORECASE)


def extract_amount(text: str) -> float | None:
    m = _MILLION.search(text)
    if m:
        return float(m.group("num").replace(",", ".")) * 1_000_000
    for m in _AMOUNT.finditer(text):
        value = float(re.sub(r"[ . ]", "", m.group("num")))
        if 1990 <= value <= 2100 and not m.group("unit"):  # une année, pas un montant
            continue
        return value
    return None


def guess_bareme(text: str) -> str | None:
    low = text.lower()
    for doc_id, keywords in BAREME_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", low) for k in keywords):
            return doc_id
    return None


def rule_based_analysis(question: str) -> QueryAnalysis:
    """Analyse sans modèle : sert de base et de repli."""
    years = [y for y in extract_years(question) if 2000 <= y <= 2100]
    articles = extract_article_refs(question)
    montant = extract_amount(question)
    bareme = guess_bareme(question)
    low = question.lower()
    if montant and bareme and re.search(r"salaire|combien|calcul|montant|retenue|payer|net|brut", low):
        intention: Intention = "calcul_bareme"
    elif len(years) >= 2 or re.search(r"\b(évolution|comparer|comparaison|par rapport à)\b", low):
        intention = "comparaison"
    elif re.search(r"\b(nouveau|nouvelle|nouveaut|changement|modifi|réforme)", low):
        intention = "nouveautes"
    else:
        intention = "information"
    return QueryAnalysis(
        question_reformulee=question.strip(),
        intention=intention,
        exercices=years,
        articles=articles,
        bareme=bareme if intention == "calcul_bareme" else None,
        montant=montant,
    )


def merge_analyses(rules: QueryAnalysis, llm: QueryAnalysis, allowed_text: str | None = None) -> QueryAnalysis:
    """Fusionne l'analyse par règles et celle du modèle.

    allowed_text : texte dont le modèle peut reprendre années et articles
    (la question, plus les questions précédentes de la conversation).
    """
    # Le modèle ne peut ajouter que des articles dont le numéro figure dans la conversation :
    # un article proposé de mémoire serait forcé dans le contexte à tort.
    allowed = allowed_text or rules.question_reformulee
    articles = list(rules.articles)
    for a in llm.articles:
        norm = parse_article_id(a)
        number = re.sub(r"\D", "", norm or "")
        if norm and norm not in articles and number and re.search(rf"\b{number}\b", allowed):
            articles.append(norm)
    # Les années viennent du texte : le modèle ne peut pas en ajouter qui n'y figurent pas.
    exercices = rules.exercices or [y for y in llm.exercices if str(y) in allowed]
    montant = rules.montant if rules.montant is not None else llm.montant
    intention = llm.intention
    bareme = llm.bareme or rules.bareme
    if intention == "calcul_bareme" and (montant is None or bareme is None):
        intention = "information"
    return QueryAnalysis(
        question_reformulee=llm.question_reformulee or rules.question_reformulee,
        intention=intention,
        exercices=sorted(set(exercices)),
        articles=articles,
        impots=llm.impots,
        bareme=bareme if intention == "calcul_bareme" else None,
        montant=montant,
    )


HISTORY_PROMPT = """La question peut être une relance d'une conversation en cours ("et pour 2025 ?", \
"et si mon salaire est de 500 000 ?"). Dans ce cas, la question reformulée doit être autonome : reprends \
de l'historique le sujet, l'impôt, l'article, l'année ou le montant nécessaires. Si la question change \
de sujet, ignore l'historique."""


def format_history(history: list) -> str:
    lines = []
    for t in history:
        lines.append(f"Utilisateur : {t.question_reformulee or t.question}")
        answer = t.answer if len(t.answer) <= 500 else t.answer[:500] + "…"
        lines.append(f"Assistant : {answer}")
    return "\n".join(lines)


def analyze_question(
    question: str,
    use_llm: bool = True,
    history: list | None = None,
    usage: LlmUsage | None = None,
) -> QueryAnalysis:
    """history : échanges précédents (objets avec question, question_reformulee, answer)."""
    rules = rule_based_analysis(question)
    if not use_llm:
        return rules

    system = ANALYSIS_PROMPT
    user = question
    allowed = question
    if history:
        system = f"{ANALYSIS_PROMPT}\n\n{HISTORY_PROMPT}"
        user = f"HISTORIQUE :\n{format_history(history)}\n\nNOUVELLE QUESTION : {question}"
        allowed = " ".join([question] + [t.question_reformulee or t.question for t in history])

    try:
        from rag_fisc_v2.clients import get_mistral

        response = get_mistral().chat.parse(
            response_format=QueryAnalysis,
            model=get_settings().analysis_model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if usage is not None:
            usage.add(response.usage)
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("réponse non structurée")
        if history and rules.montant is None and parsed.montant:
            # Montant repris d'une question précédente, s'il y figure vraiment.
            previous = [extract_amount(t.question) for t in history]
            if parsed.montant in previous:
                rules.montant = parsed.montant
        return merge_analyses(rules, parsed, allowed_text=allowed)
    except Exception as exc:  # le modèle est une amélioration, pas une dépendance
        log.warning("Analyse par le modèle indisponible (%s), repli sur les règles", exc)
        return rules
