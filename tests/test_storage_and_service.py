import time

from rag_fisc_v2 import storage
from rag_fisc_v2.config import get_settings
from rag_fisc_v2.domain import Canal, Profil
from rag_fisc_v2.generation.generator import Generation
from rag_fisc_v2.retrieval.rerank import AutoReranker, MistralReranker, PineconeReranker
from rag_fisc_v2.retrieval.retriever import Passage, RetrievalResult
from rag_fisc_v2.schemas.api import AskRequest
from rag_fisc_v2.services import rag as rag_module
from rag_fisc_v2.services.rag import RagService


def test_hash_is_stable_and_not_the_number():
    h = storage.hash_user("whatsapp:237600000001")
    assert h == storage.hash_user("whatsapp:237600000001")
    assert "237600000001" not in h


def test_dedupe():
    assert storage.mark_message_processed("m1") is True
    assert storage.mark_message_processed("m1") is False


def test_history_order_limit_and_expiry():
    s = get_settings()
    for i in range(6):
        storage.add_turn("u", f"q{i}", f"r{i}")
    history = storage.get_history("u")
    assert [t.question for t in history] == ["q2", "q3", "q4", "q5"][-s.history_max_turns :]
    storage._run("UPDATE turns SET created_at = ?", (time.time() - 10 * 3600,))
    assert storage.get_history("u") == []


def test_backend_is_sqlite_in_tests():
    assert storage.backend_name() == "sqlite"


def test_unknown_interaction_column_rejected():
    import pytest

    with pytest.raises(ValueError):
        storage.log_interaction(canal="api", user_hash="u", question="q", colonne_pirate="x")


def test_counters():
    assert storage.increment("x") == 1
    assert storage.increment("x") == 2
    assert storage.get_counter("x") == 2
    assert storage.get_counter("x", "month") == 0


def test_auto_reranker_switches_to_mistral_after_budget(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "rerank_provider", "auto")
    monkeypatch.setattr(s, "pinecone_rerank_monthly_budget", 2)
    auto = AutoReranker()
    assert isinstance(auto._primary(), PineconeReranker)
    storage.increment("rerank_pinecone", "month")
    storage.increment("rerank_pinecone", "month")
    assert isinstance(auto._primary(), MistralReranker)


def test_auto_reranker_falls_back_to_vector_order(monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("quota")

    monkeypatch.setattr(PineconeReranker, "rerank", boom)
    monkeypatch.setattr(MistralReranker, "rerank", boom)
    auto = AutoReranker()
    passages = [Passage("a", "x", {"type": "cgi"}, 0.2), Passage("b", "y", {"type": "cgi"}, 0.9)]
    ranked = auto.rerank("q", passages, top_n=2)
    assert [p.node_id for p in ranked] == ["b", "a"]
    assert auto.last_provider == "aucun"


# --- Service : mémoire de conversation et suivi d'usage ------------------------------------


class FakeRetriever:
    def __init__(self):
        self.analyses = []

    def retrieve(self, analysis):
        self.analyses.append(analysis)
        p = Passage("cgi-2026#1", "Article 17.- Taux 30 %", {"doc_id": "cgi-2026", "type": "cgi", "exercice": 2026, "article": "17", "titre_document": "CGI"}, 0.9)
        return RetrievalResult(passages=[p], exercices=[2026], reranker="pinecone")


def test_service_keeps_history_and_logs(monkeypatch):
    seen_history = []

    def fake_analyze(question, use_llm=True, history=None, usage=None):
        seen_history.append(list(history or []))
        from rag_fisc_v2.retrieval.analysis import rule_based_analysis

        return rule_based_analysis(question)

    def fake_generate(**kwargs):
        kwargs["usage"].prompt_tokens += 100
        return Generation(text="30 % [S1]", cited=["S1"]), {"S1": FakeRetriever().retrieve(None).passages[0]}

    monkeypatch.setattr(rag_module, "analyze_question", fake_analyze)
    monkeypatch.setattr(rag_module, "generate_answer", fake_generate)
    service = RagService(manifest=None, retriever=FakeRetriever())

    r1 = service.ask(AskRequest(question="Quel est le taux de l'IS ?", conversation_id="abc"), canal=Canal.WHATSAPP)
    service.ask(AskRequest(question="Et pour 2025 ?", conversation_id="abc"), canal=Canal.WHATSAPP)

    assert seen_history[0] == []
    assert seen_history[1][0].question == "Quel est le taux de l'IS ?"
    assert r1.interaction_id is not None
    stats = storage.usage_stats()
    assert stats["questions"] == 2
    assert stats["utilisateurs"] == 1
    assert stats["par_canal"] == {"whatsapp": 2}
    assert stats["tokens_entree"] == 200


def test_service_without_tracking_writes_nothing(monkeypatch):
    from rag_fisc_v2.retrieval.analysis import rule_based_analysis

    monkeypatch.setattr(rag_module, "analyze_question", lambda q, **k: rule_based_analysis(q))
    monkeypatch.setattr(
        rag_module, "generate_answer", lambda **k: (Generation(text="ok", cited=[]), {})
    )
    RagService(manifest=None, retriever=FakeRetriever(), track=False).ask(
        AskRequest(question="Quel est le taux de l'IS ?", profil=Profil.EXPERT, conversation_id="z")
    )
    assert storage.usage_stats()["questions"] == 0


def test_bareme_result_always_shown(monkeypatch):
    """Si le modèle ne cite pas le barème, le service ajoute lui-même la ligne exacte."""
    from rag_fisc_v2.ingestion.baremes import Bareme
    from rag_fisc_v2.retrieval.analysis import QueryAnalysis

    table = Bareme({
        "doc_id": "bareme-tdl", "titre": "Barème TDL", "exercice": None, "colonnes": ["base", "tdl"],
        "libelles": {"base": "Base", "tdl": "Taxe de développement local (TDL)"}, "pas": 250,
        "lignes": [[999750, 2500], [1000000, 2500]],
    })
    analysis = QueryAnalysis(question_reformulee="TDL sur 1 000 000", intention="calcul_bareme", bareme="bareme-tdl", montant=1_000_000)
    monkeypatch.setattr(rag_module, "analyze_question", lambda q, **k: analysis)
    monkeypatch.setattr(rag_module, "get_bareme", lambda doc_id: table)
    monkeypatch.setattr(
        rag_module, "generate_answer", lambda **k: (Generation(text="La TDL annuelle est de 30 000 FCFA [S1].", cited=["S1"]), {"S1": FakeRetriever().retrieve(None).passages[0]})
    )
    response = RagService(manifest=None, retriever=FakeRetriever(), track=False).ask(AskRequest(question="TDL sur 1 000 000 FCFA ?"))
    assert response.reponse.startswith("Selon le barème mensuel officiel, pour un salaire de 1 000 000 FCFA : Taxe de développement local (TDL) : 2 500 FCFA. [B]")
    assert "30 000 FCFA" in response.reponse
