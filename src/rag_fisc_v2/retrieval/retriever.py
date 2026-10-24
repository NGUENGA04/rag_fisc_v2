"""Recherche des passages pertinents pour une question analysée.

Pour chaque exercice visé :
  1. recherche exacte des articles du CGI cités dans la question ;
  2. recherche sémantique filtrée par exercice ;
  3. ajout des passages de loi de finances, circulaire ou arrêté qui citent
     les articles du CGI trouvés ;
  4. reranking multilingue, en gardant toujours les articles demandés explicitement.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from rag_fisc_v2.config import get_settings
from rag_fisc_v2.domain import DOC_TYPE_AUTHORITY, EXERCICE_NON_DATE, DocType
from rag_fisc_v2.ingestion.manifest import Manifest
from rag_fisc_v2.ingestion.references import LATIN_SUFFIXES
from rag_fisc_v2.retrieval.analysis import QueryAnalysis

log = logging.getLogger(__name__)

LINKED_TYPES = [DocType.LFI.value, DocType.CIRCULAIRE.value, DocType.ARRETE.value]

# Articles à joindre d'office : les centimes additionnels communaux (CAC) s'ajoutent au principal
# de ces impôts. Sans eux, le modèle donne un taux incomplet (TVA à 17,5 % au lieu de 19,25 %).
_CAC_ARTICLES = ["C82", "C83"]
_CAC_TAXES = re.compile(
    r"\b(tva|taxe sur la valeur ajout[ée]e|is|imp[ôo]t sur les soci[ée]t[ée]s|irpp|imp[ôo]t sur le revenu"
    r"|droits? d.accises?|accises?)\b",
    re.IGNORECASE,
)
_AMOUNT_OR_RATE = re.compile(r"\b(taux|combien|calcul\w*|montant|payer|total|co[ûu]te?|pourcentage|%)", re.IGNORECASE)


def companion_articles(analysis: QueryAnalysis) -> list[str]:
    """Articles du CGI à fournir d'office pour que la réponse soit complète."""
    text = f"{analysis.question_reformulee} {' '.join(analysis.impots)}"
    if _CAC_TAXES.search(text) and _AMOUNT_OR_RATE.search(text):
        return [a for a in _CAC_ARTICLES if a not in analysis.articles]
    return []


# Borne de sécurité sur la taille du filtre $in (6 articles x 11 variantes = 66 au plus).
_MAX_FAMILY = 120


def article_family(articles: list[str]) -> list[str]:
    """Articles de base et variantes : "17" ou "17 quater" -> 17, 17 bis, 17 ter, ... 17 decies.

    Dans le CGI, un article "bis" ou "ter" complète l'article de base ou y déroge :
    il faut les lire ensemble.
    """
    family: list[str] = []
    for article in articles:
        base = article.split(" ", 1)[0]
        for variant in [base, *(f"{base} {s}" for s in LATIN_SUFFIXES)]:
            if variant not in family:
                family.append(variant)
    return family[:_MAX_FAMILY]


@dataclass
class Passage:
    node_id: str
    text: str
    metadata: dict
    score: float = 0.0
    origins: set[str] = field(default_factory=set)
    exercice_cible: int | None = None

    @property
    def doc_type(self) -> DocType:
        return DocType(self.metadata.get("type", "fiche"))

    @property
    def article(self) -> str | None:
        return self.metadata.get("article")

    def label(self) -> str:
        m = self.metadata
        parts = [m.get("titre_document", m.get("doc_id", ""))]
        if m.get("article"):
            prefix = "article" if self.doc_type != DocType.LFI else "article de la loi"
            parts.append(f"{prefix} {m['article']}")
        if self.doc_type == DocType.LFI and m.get("articles_cites"):
            parts.append(f"modifiant ou citant l'article {m['articles_cites'][0]} du CGI")
        pages = (m.get("page_debut"), m.get("page_fin"))
        if pages[0]:
            parts.append(f"p. {pages[0]}" if pages[0] == pages[1] else f"p. {pages[0]}-{pages[1]}")
        return ", ".join(str(p) for p in parts if p)


@dataclass
class RetrievalResult:
    passages: list[Passage]
    exercices: list[int]
    avertissements: list[str] = field(default_factory=list)
    reranker: str = "aucun"


class FiscalRetriever:
    def __init__(self, manifest: Manifest, vector_store=None, embed_model=None, reranker=None):
        self.manifest = manifest
        self.settings = get_settings()
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._reranker = reranker

    # --- dépendances paresseuses (permettent les tests sans clé) ------------------

    @property
    def vector_store(self):
        if self._vector_store is None:
            from rag_fisc_v2.clients import get_vector_store

            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def embed_model(self):
        if self._embed_model is None:
            from rag_fisc_v2.clients import get_embed_model

            self._embed_model = get_embed_model()
        return self._embed_model

    @property
    def reranker(self):
        if self._reranker is None:
            from rag_fisc_v2.retrieval.rerank import AutoReranker

            self._reranker = AutoReranker()
        return self._reranker

    # --- requêtes ---------------------------------------------------------------------

    def _query(self, vector: list[float], top_k: int, flt: dict, origin: str, year: int) -> list[Passage]:
        result = self.vector_store.query(
            VectorStoreQuery(query_embedding=vector, similarity_top_k=top_k),
            pinecone_query_filters=flt,
            include_values=False,
        )
        passages = []
        for node, score in zip(result.nodes or [], result.similarities or [], strict=False):
            assert isinstance(node, TextNode)
            passages.append(
                Passage(
                    node_id=node.node_id,
                    text=node.get_content(),
                    metadata=dict(node.metadata),
                    score=float(score),
                    origins={origin},
                    exercice_cible=year,
                )
            )
        return passages

    def resolve_exercices(self, analysis: QueryAnalysis) -> tuple[list[int], list[str]]:
        """Exercices à interroger, et avertissements sur la couverture du corpus."""
        warnings: list[str] = []
        available = self.manifest.exercices()
        years = analysis.exercices or [self.manifest.latest_exercice()]
        if analysis.intention == "comparaison" and len(years) == 1:
            # "Qu'est-ce qui a changé en 2026" : comparer avec l'exercice précédent disponible.
            previous = [y for y in available if y < years[0]]
            if previous:
                years = [previous[-1], years[0]]
        for y in years:
            if y not in available:
                warnings.append(
                    f"Le corpus ne contient aucun document pour l'exercice {y}. "
                    f"Exercices couverts : {', '.join(map(str, available))}."
                )
        return sorted(set(years)), warnings

    def year_filter(self, year: int) -> tuple[dict, str | None, str | None]:
        """Filtre Pinecone pour un exercice, le CGI applicable et un éventuel avertissement."""
        cgi = self.manifest.cgi_for(year)
        warning = None
        if cgi is None:
            cgi = self.manifest.latest_cgi()
            if cgi is not None:
                warning = (
                    f"Le CGI de l'exercice {year} n'est pas dans le corpus. Les articles cités proviennent "
                    f"du {cgi.titre} : ils ont pu être modifiés depuis {year}. La loi de finances et la "
                    f"circulaire de {year} font foi pour cet exercice."
                )
        conditions: list[dict] = [{"exercice": year}, {"exercice": EXERCICE_NON_DATE}]
        if cgi is not None and cgi.exercice != year:
            conditions.append({"doc_id": cgi.doc_id})
        return {"$or": conditions}, (cgi.doc_id if cgi else None), warning

    def retrieve_for_year(self, analysis: QueryAnalysis, vector: list[float], year: int) -> tuple[list[Passage], str | None]:
        s = self.settings
        flt, cgi_id, warning = self.year_filter(year)

        # Requêtes indépendantes : lancées en parallèle pour réduire la latence.
        queries: dict[str, tuple] = {"dense": (s.top_k_dense, flt, "dense")}
        if analysis.articles and cgi_id:
            queries["article"] = (s.top_k_article, {"doc_id": cgi_id, "article": {"$in": analysis.articles}}, "article")
            queries["lien"] = (
                s.top_k_linked,
                {"exercice": year, "type": {"$in": LINKED_TYPES}, "articles_cites": {"$in": analysis.articles}},
                "lien",
            )
        if analysis.intention == "nouveautes":
            queries["nouveautes"] = (s.top_k_dense, {"exercice": year, "type": {"$in": LINKED_TYPES}}, "dense")
        companions = companion_articles(analysis)
        if companions and cgi_id:
            queries["associe"] = (len(companions) * 2, {"doc_id": cgi_id, "article": {"$in": companions}}, "associe")

        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            futures = {
                name: pool.submit(self._query, vector, top_k, query_filter, origin, year)
                for name, (top_k, query_filter, origin) in queries.items()
            }
            results = {name: future.result() for name, future in futures.items()}

        dense = results["dense"]
        found: list[Passage] = [
            p for name in ("article", "associe", "lien", "nouveautes") for p in results.get(name, [])
        ]
        found += dense

        # Articles du CGI parmi les meilleurs résultats : on remonte leurs commentaires
        # (lois de finances, circulaires) et leur famille (17 -> 17 bis, 17 ter...).
        top_articles = []
        for p in dense[:6]:
            if p.doc_type == DocType.CGI and p.article and p.article not in top_articles:
                top_articles.append(p.article)
        new_links = [a for a in top_articles if a not in analysis.articles]
        family = article_family(top_articles + analysis.articles)

        follow_up: dict[str, tuple] = {}
        if new_links:
            follow_up["lien"] = (
                s.top_k_linked,
                {"exercice": year, "type": {"$in": LINKED_TYPES}, "articles_cites": {"$in": new_links}},
                "lien",
            )
        if family and cgi_id:
            follow_up["famille"] = (s.top_k_family, {"doc_id": cgi_id, "article": {"$in": family}}, "famille")
        if follow_up:
            with ThreadPoolExecutor(max_workers=len(follow_up)) as pool:
                futures = [
                    pool.submit(self._query, vector, top_k, query_filter, origin, year)
                    for top_k, query_filter, origin in follow_up.values()
                ]
                for future in futures:
                    found += future.result()
        return found, warning

    @staticmethod
    def merge(passages: list[Passage]) -> list[Passage]:
        by_id: dict[str, Passage] = {}
        for p in passages:
            if p.node_id in by_id:
                kept = by_id[p.node_id]
                kept.score = max(kept.score, p.score)
                kept.origins |= p.origins
            else:
                by_id[p.node_id] = p
        return list(by_id.values())

    def retrieve(self, analysis: QueryAnalysis) -> RetrievalResult:
        years, warnings = self.resolve_exercices(analysis)
        vector = self.embed_model.get_query_embedding(analysis.question_reformulee)
        s = self.settings

        selected: list[Passage] = []
        reranker_used = "aucun"
        per_year_n = s.rerank_top_n if len(years) == 1 else max(4, s.rerank_top_n // len(years) + 1)
        for year in years:
            candidates, warning = self.retrieve_for_year(analysis, vector, year)
            if warning and warning not in warnings:
                warnings.append(warning)
            candidates = self.merge(candidates)
            if not candidates:
                continue
            ranked = self.reranker.rerank(analysis.question_reformulee, candidates, top_n=per_year_n)
            min_score = getattr(self.reranker, "min_score", s.rerank_min_score)
            ranked = [p for p in ranked if p.score >= min_score]
            reranker_used = getattr(self.reranker, "last_provider", "inconnu")
            # Les articles explicitement demandés restent toujours dans le contexte.
            pinned = [p for p in candidates if "article" in p.origins and p.doc_type == DocType.CGI]
            for p in pinned[:3]:
                p.origins.add("demande")
                if all(p.node_id != r.node_id for r in ranked):
                    ranked.append(p)
            # Articles joints d'office (CAC) : conservés même si le reranker les écarte.
            for p in [p for p in candidates if "associe" in p.origins][:2]:
                if all(p.node_id != r.node_id for r in ranked):
                    ranked.append(p)
            selected += ranked

        selected = self.merge(selected)
        selected.sort(key=self._final_order)
        return RetrievalResult(passages=selected, exercices=years, avertissements=warnings, reranker=reranker_used)

    @staticmethod
    def _final_order(p: Passage) -> tuple:
        """Articles demandés d'abord, puis passages reranqués par score, puis le reste.

        Les scores du reranker et ceux de la recherche vectorielle n'ont pas la même
        échelle : on ne les compare jamais entre eux.
        """
        if "demande" in p.origins:
            group = 0
        elif "rerank" in p.origins:
            group = 1
        else:
            group = 2
        return (group, -p.score, DOC_TYPE_AUTHORITY[p.doc_type])
