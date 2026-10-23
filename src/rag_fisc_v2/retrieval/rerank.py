"""Reranking des passages.

Trois implémentations, choisies par RERANK_PROVIDER :
- pinecone : bge-reranker-v2-m3 hébergé par Pinecone (500 appels par mois en offre gratuite) ;
- mistral  : Mistral Small note la pertinence de chaque passage ;
- auto     : Pinecone tant que le budget mensuel n'est pas atteint, puis Mistral.
En cas d'erreur, l'ordre de la recherche vectorielle est conservé.
"""

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from rag_fisc_v2.config import get_settings

if TYPE_CHECKING:
    from rag_fisc_v2.retrieval.retriever import Passage

log = logging.getLogger(__name__)

# bge-reranker-v2-m3 accepte environ 1 024 jetons par paire question + passage.
_MAX_DOC_CHARS = 3000
_MISTRAL_DOC_CHARS = 1200
_MISTRAL_MAX_CANDIDATES = 30


def rerank_text(p: "Passage", limit: int = _MAX_DOC_CHARS) -> str:
    m = p.metadata
    header = " | ".join(str(x) for x in (m.get("titre_document"), m.get("hierarchie"), m.get("article")) if x)
    return f"{header}\n{p.text}"[:limit]


def _vector_order(passages: list["Passage"], top_n: int) -> list["Passage"]:
    return sorted(passages, key=lambda p: -p.score)[:top_n]


class PineconeReranker:
    name = "pinecone"

    def __init__(self, model: str | None = None):
        self.model = model or get_settings().rerank_model
        self.min_score = get_settings().rerank_min_score

    def rerank(self, query: str, passages: list["Passage"], top_n: int) -> list["Passage"]:
        if not passages:
            return []
        from rag_fisc_v2.clients import get_pinecone
        from rag_fisc_v2.storage import increment

        increment("rerank_pinecone", "month")
        result = get_pinecone().inference.rerank(
            model=self.model,
            query=query,
            documents=[{"id": p.node_id, "text": rerank_text(p)} for p in passages],
            rank_fields=["text"],
            top_n=min(top_n, len(passages)),
            return_documents=False,
            parameters={"truncate": "END"},
        )
        ranked = []
        for item in result.data:
            p = passages[item.index]
            p.score = float(item.score)
            p.origins.add("rerank")
            ranked.append(p)
        return ranked


class _Score(BaseModel):
    id: str = Field(description="Identifiant du passage, ex. P3")
    score: int = Field(ge=0, le=10, description="0 = sans rapport, 10 = répond directement à la question")


class _Scores(BaseModel):
    scores: list[_Score]


_MISTRAL_PROMPT = """Tu évalues la pertinence de passages de textes fiscaux camerounais pour répondre à une question.
Note chaque passage de 0 à 10 :
10 = contient directement la réponse ; 6 = utile (condition, exception, définition liée) ;
3 = même sujet mais ne répond pas ; 0 = sans rapport.
Note tous les passages, sans en oublier."""


class MistralReranker:
    name = "mistral"

    def __init__(self):
        self.min_score = get_settings().mistral_rerank_min_score

    def rerank(self, query: str, passages: list["Passage"], top_n: int) -> list["Passage"]:
        if not passages:
            return []
        from rag_fisc_v2.clients import get_mistral

        candidates = _vector_order(passages, _MISTRAL_MAX_CANDIDATES)
        listing = "\n\n".join(
            f"[P{i}] {rerank_text(p, _MISTRAL_DOC_CHARS)}" for i, p in enumerate(candidates)
        )
        response = get_mistral().chat.parse(
            response_format=_Scores,
            model=get_settings().analysis_model,
            temperature=0,
            messages=[
                {"role": "system", "content": _MISTRAL_PROMPT},
                {"role": "user", "content": f"QUESTION : {query}\n\nPASSAGES :\n{listing}"},
            ],
        )
        parsed = response.choices[0].message.parsed
        scores = {s.id.strip("[] ").upper(): s.score for s in (parsed.scores if parsed else [])}
        for i, p in enumerate(candidates):
            p.score = scores.get(f"P{i}", 0) / 10
            p.origins.add("rerank")
        return sorted(candidates, key=lambda p: -p.score)[:top_n]


class AutoReranker:
    """Choisit le reranker selon la configuration et le budget Pinecone du mois."""

    def __init__(self):
        self.settings = get_settings()
        self.last_provider = "aucun"
        self.min_score = self.settings.rerank_min_score

    def _primary(self):
        provider = self.settings.rerank_provider
        if provider == "pinecone":
            return PineconeReranker()
        if provider == "mistral":
            return MistralReranker()
        if provider == "none":
            return None
        from rag_fisc_v2.storage import get_counter

        if get_counter("rerank_pinecone", "month") < self.settings.pinecone_rerank_monthly_budget:
            return PineconeReranker()
        return MistralReranker()

    def rerank(self, query: str, passages: list["Passage"], top_n: int) -> list["Passage"]:
        reranker = self._primary()
        if reranker is None or not passages:
            self.last_provider = "aucun"
            self.min_score = float("-inf")
            return _vector_order(passages, top_n)
        chain = [reranker]
        if self.settings.rerank_provider == "auto" and isinstance(reranker, PineconeReranker):
            chain.append(MistralReranker())  # quota Pinecone épuisé plus tôt que prévu, par exemple
        for candidate in chain:
            try:
                ranked = candidate.rerank(query, passages, top_n)
                self.last_provider = candidate.name
                self.min_score = candidate.min_score
                return ranked
            except Exception as exc:
                log.warning("Reranking %s indisponible (%s)", candidate.name, exc)
        log.warning("Aucun reranking disponible, ordre vectoriel conservé")
        self.last_provider = "aucun"
        self.min_score = float("-inf")
        return _vector_order(passages, top_n)
