"""Tests de la recherche avec un faux vector store, sans appel réseau."""

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from rag_fisc_v2.ingestion.manifest import Manifest
from rag_fisc_v2.retrieval.analysis import QueryAnalysis
from rag_fisc_v2.retrieval.retriever import FiscalRetriever

MANIFEST = Manifest.model_validate(
    {
        "documents": [
            {"doc_id": "cgi-2026", "fichier": "a.pdf", "type": "cgi", "exercice": 2026, "titre": "CGI 2026"},
            {"doc_id": "lfi-2025", "fichier": "b.pdf", "type": "lfi", "exercice": 2025, "titre": "LFI 2025"},
            {"doc_id": "lfi-2026", "fichier": "c.pdf", "type": "lfi", "exercice": 2026, "titre": "LFI 2026"},
            {"doc_id": "circ-2026", "fichier": "d.pdf", "type": "circulaire", "exercice": 2026, "titre": "Circ 2026"},
        ]
    }
)

NODES = [
    TextNode(id_="cgi-2026#1", text="Article 17.- Taux IS 30 %", metadata={"doc_id": "cgi-2026", "type": "cgi", "exercice": 2026, "article": "17"}),
    TextNode(id_="cgi-2026#2", text="Article 21.- Acomptes", metadata={"doc_id": "cgi-2026", "type": "cgi", "exercice": 2026, "article": "21"}),
    TextNode(id_="circ-2026#1", text="Commentaire de l'article 17", metadata={"doc_id": "circ-2026", "type": "circulaire", "exercice": 2026, "articles_cites": ["17"]}),
    TextNode(id_="lfi-2025#1", text="Modification de l'article 17", metadata={"doc_id": "lfi-2025", "type": "lfi", "exercice": 2025, "article": "1", "articles_cites": ["17"]}),
]


def _matches(meta: dict, flt: dict) -> bool:
    for key, cond in flt.items():
        if key == "$or":
            if not any(_matches(meta, c) for c in cond):
                return False
            continue
        value = meta.get(key)
        if isinstance(cond, dict) and "$in" in cond:
            values = value if isinstance(value, list) else [value]
            if not set(values) & set(cond["$in"]):
                return False
        elif value != cond:
            return False
    return True


class FakeVectorStore:
    def __init__(self):
        self.filters = []

    def query(self, query, pinecone_query_filters, **_):
        self.filters.append(pinecone_query_filters)
        hits = [n for n in NODES if _matches(n.metadata, pinecone_query_filters)][: query.similarity_top_k]
        return VectorStoreQueryResult(nodes=hits, similarities=[0.8] * len(hits), ids=[n.node_id for n in hits])


class FakeEmbed:
    def get_query_embedding(self, text):
        return [0.0] * 4


class FakeReranker:
    def rerank(self, query, passages, top_n):
        return passages[:top_n]


def _retriever():
    return FiscalRetriever(MANIFEST, vector_store=FakeVectorStore(), embed_model=FakeEmbed(), reranker=FakeReranker())


def test_default_year_is_latest():
    r = _retriever()
    result = r.retrieve(QueryAnalysis(question_reformulee="taux IS", intention="information"))
    assert result.exercices == [2026]
    assert {p.metadata["doc_id"] for p in result.passages} <= {"cgi-2026", "circ-2026", "lfi-2026"}


def test_explicit_article_and_linked_commentary():
    r = _retriever()
    result = r.retrieve(QueryAnalysis(question_reformulee="article 17", intention="information", articles=["17"]))
    ids = [p.node_id for p in result.passages]
    assert "cgi-2026#1" in ids
    assert "circ-2026#1" in ids
    exact = next(p for p in result.passages if p.node_id == "cgi-2026#1")
    assert "article" in exact.origins


def test_missing_cgi_year_uses_latest_with_warning():
    r = _retriever()
    result = r.retrieve(QueryAnalysis(question_reformulee="IS 2025", intention="information", exercices=[2025], articles=["17"]))
    assert any("CGI de l'exercice 2025" in w for w in result.avertissements)
    ids = {p.node_id for p in result.passages}
    assert "lfi-2025#1" in ids  # la loi de finances 2025 est retrouvée
    assert "cgi-2026#1" in ids  # avec l'article du CGI le plus récent, signalé


def test_unknown_year_warns():
    r = _retriever()
    result = r.retrieve(QueryAnalysis(question_reformulee="x", intention="information", exercices=[2019]))
    assert any("aucun document pour l'exercice 2019" in w for w in result.avertissements)


def test_comparison_adds_previous_year():
    r = _retriever()
    years, _ = r.resolve_exercices(QueryAnalysis(question_reformulee="x", intention="comparaison", exercices=[2026]))
    assert years == [2025, 2026]


def test_requested_article_first_and_scores_not_mixed():
    class ScoringReranker:
        """Donne un score de reranking bas à tout, sauf au commentaire de circulaire."""

        def rerank(self, query, passages, top_n):
            for p in passages:
                p.score = 0.9 if p.node_id == "circ-2026#1" else 0.1
                p.origins.add("rerank")
            return sorted(passages, key=lambda p: -p.score)[:1]  # l'article 17 n'est pas retenu

    r = FiscalRetriever(MANIFEST, vector_store=FakeVectorStore(), embed_model=FakeEmbed(), reranker=ScoringReranker())
    result = r.retrieve(QueryAnalysis(question_reformulee="article 17", intention="information", articles=["17"]))
    ids = [p.node_id for p in result.passages]
    assert ids[0] == "cgi-2026#1"  # article demandé, réinséré et placé en tête
    assert ids[1] == "circ-2026#1"
