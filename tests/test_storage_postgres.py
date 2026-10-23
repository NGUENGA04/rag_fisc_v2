"""Tests du moteur PostgreSQL. Ignorés sauf si TEST_DATABASE_URL est défini.

Exemple : TEST_DATABASE_URL=postgresql://test@localhost:55432/postgres uv run pytest tests/test_storage_postgres.py
La base indiquée est vidée des tables du projet avant les tests : n'utilisez jamais la base de production.
"""

import os
import time

import pytest
from pydantic import SecretStr

from rag_fisc_v2 import storage
from rag_fisc_v2.config import get_settings

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL non défini")


@pytest.fixture(autouse=True)
def postgres(isolated_db):
    import psycopg

    with psycopg.connect(URL, autocommit=True) as conn:
        for table in ("processed_messages", "turns", "user_prefs", "counters", "interactions"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    settings = get_settings()
    settings.database_url = SecretStr(URL)
    storage._backends.clear()
    yield
    storage._backends.clear()


def test_backend_and_schema():
    assert storage.backend_name() == "postgres"


def test_dedupe_history_counters():
    assert storage.mark_message_processed("m1") is True
    assert storage.mark_message_processed("m1") is False

    for i in range(3):
        storage.add_turn("u", f"q{i}", f"r{i}", f"q{i} reformulée")
    history = storage.get_history("u")
    assert [t.question for t in history] == ["q0", "q1", "q2"]
    assert history[0].question_reformulee == "q0 reformulée"
    storage.clear_history("u")
    assert storage.get_history("u") == []

    assert storage.increment("c") == 1
    assert storage.increment("c") == 2
    assert storage.get_counter("c") == 2


def test_prefs_upsert():
    assert storage.get_profil("u") is None
    storage.set_profil("u", "grand_public")
    storage.set_profil("u", "expert")
    assert storage.get_profil("u") == "expert"


def test_interactions_feedback_and_stats():
    i1 = storage.log_interaction(
        canal="whatsapp", user_hash="u1", question="q", intention="information",
        exercices=[2026], latence_ms=1200, tokens_entree=100, tokens_sortie=50,
    )
    storage.log_interaction(canal="api", user_hash="u2", question="q2", erreur="boom", created_at=time.time())
    assert storage.set_feedback(i1, 1, user_hash="autre") is False
    assert storage.set_feedback(i1, 1, user_hash="u1") is True

    stats = storage.usage_stats(7)
    assert stats["stockage"] == "postgres"
    assert stats["questions"] == 2
    assert stats["utilisateurs"] == 2
    assert stats["erreurs"] == 1
    assert stats["avis_positifs"] == 1
    assert stats["tokens_entree"] == 100
    assert stats["par_canal"] == {"whatsapp": 1, "api": 1}
    assert len(stats["par_jour"]) == 1 and stats["par_jour"][0]["n"] == 2


def test_reconnects_after_connection_loss():
    storage.increment("r")
    backend = storage._db()
    backend._connection.close()
    assert storage.increment("r") == 2
