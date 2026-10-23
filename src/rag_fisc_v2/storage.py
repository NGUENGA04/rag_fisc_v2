"""Stockage applicatif : conversations, préférences, compteurs et suivi d'usage.

Deux moteurs, choisis automatiquement :
- PostgreSQL (Neon, ou tout autre) si DATABASE_URL est défini. Obligatoire sur
  Vercel, dont le disque n'est ni persistant ni partagé entre instances.
- SQLite sinon (développement local, Docker avec volume).

Les requêtes sont écrites une seule fois, avec des marqueurs "?" convertis pour
PostgreSQL. Les utilisateurs sont identifiés par un hachage salé de leur
identifiant : aucun numéro de téléphone n'est stocké en clair.
"""

import hashlib
import hmac
import json
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_fisc_v2.config import get_settings

_TABLES = """
CREATE TABLE IF NOT EXISTS processed_messages (
    message_id TEXT PRIMARY KEY,
    received_at {float} NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id {serial},
    user_hash TEXT NOT NULL,
    created_at {float} NOT NULL,
    question TEXT NOT NULL,
    question_reformulee TEXT,
    answer TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_hash, created_at);
CREATE TABLE IF NOT EXISTS user_prefs (
    user_hash TEXT PRIMARY KEY,
    profil TEXT NOT NULL,
    first_seen {float} NOT NULL,
    updated_at {float} NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (
    name TEXT NOT NULL,
    period TEXT NOT NULL,
    value INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (name, period)
);
CREATE TABLE IF NOT EXISTS interactions (
    id {serial},
    created_at {float} NOT NULL,
    canal TEXT NOT NULL,
    user_hash TEXT NOT NULL,
    profil TEXT,
    question TEXT NOT NULL,
    intention TEXT,
    exercices TEXT,
    nb_sources INTEGER,
    reranker TEXT,
    latence_ms INTEGER,
    tokens_entree INTEGER,
    tokens_sortie INTEGER,
    erreur TEXT,
    feedback INTEGER
);
CREATE INDEX IF NOT EXISTS idx_interactions_date ON interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_hash, created_at);
"""

_INTERACTION_COLUMNS = {
    "created_at", "canal", "user_hash", "profil", "question", "intention", "exercices", "nb_sources",
    "reranker", "latence_ms", "tokens_entree", "tokens_sortie", "erreur", "feedback",
}


class _Backend:
    """Interface commune : exécute une requête et renvoie les lignes sous forme de dict."""

    day_expr: str  # expression SQL qui donne la date (AAAA-MM-JJ) d'un horodatage epoch

    def execute(self, sql: str, params: tuple = ()) -> tuple[list[dict], int]:
        raise NotImplementedError


class _SQLite(_Backend):
    day_expr = "date(created_at, 'unixepoch')"

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_TABLES.format(float="REAL", serial="INTEGER PRIMARY KEY AUTOINCREMENT"))

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = ()) -> tuple[list[dict], int]:
        with self.lock, self._conn() as conn:
            cur = conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()] if cur.description else []
            return rows, cur.rowcount


class _Postgres(_Backend):
    day_expr = "to_char(to_timestamp(created_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD')"

    def __init__(self, url: str):
        self.url = url
        self.lock = threading.RLock()
        self._connection = None
        for statement in _TABLES.format(float="DOUBLE PRECISION", serial="BIGSERIAL PRIMARY KEY").split(";"):
            if statement.strip():
                self.execute(statement)

    def _conn(self):
        import psycopg
        from psycopg.rows import dict_row

        if self._connection is None or self._connection.closed:
            # prepare_threshold=None : compatible avec le pooler de Neon (PgBouncer en mode transaction).
            self._connection = psycopg.connect(
                self.url, autocommit=True, row_factory=dict_row, prepare_threshold=None, connect_timeout=10
            )
        return self._connection

    def execute(self, sql: str, params: tuple = ()) -> tuple[list[dict], int]:
        import psycopg

        sql = sql.replace("?", "%s")
        with self.lock:
            for attempt in (1, 2):
                try:
                    with self._conn().cursor() as cur:
                        cur.execute(sql, params)
                        rows = list(cur.fetchall()) if cur.description else []
                        return rows, cur.rowcount
                except psycopg.OperationalError:
                    # Connexion coupée (instance serverless réveillée, base en veille) : une seule reconnexion.
                    self._connection = None
                    if attempt == 2:
                        raise
        raise RuntimeError("inaccessible")


_backends: dict[str, _Backend] = {}
_backends_lock = threading.Lock()


def _db() -> _Backend:
    settings = get_settings()
    key = settings.database_url.get_secret_value() if settings.database_url else f"sqlite:{settings.app_db_path}"
    with _backends_lock:
        if key not in _backends:
            _backends[key] = _Postgres(key) if settings.database_url else _SQLite(settings.app_db_path)
        return _backends[key]


def _query(sql: str, params: tuple = ()) -> list[dict]:
    return _db().execute(sql, params)[0]


def _run(sql: str, params: tuple = ()) -> int:
    return _db().execute(sql, params)[1]


def backend_name() -> str:
    return "postgres" if isinstance(_db(), _Postgres) else "sqlite"


def hash_user(identifier: str) -> str:
    salt = get_settings().user_hash_salt.get_secret_value().encode()
    return hmac.new(salt, identifier.encode(), hashlib.sha256).hexdigest()[:32]


# --- Déduplication des messages entrants ---------------------------------------------


def mark_message_processed(message_id: str) -> bool:
    """Vrai si le message est nouveau. Faux s'il a déjà été reçu (renvoi de Meta)."""
    rows = _query(
        "INSERT INTO processed_messages(message_id, received_at) VALUES (?, ?) "
        "ON CONFLICT(message_id) DO NOTHING RETURNING message_id",
        (message_id, time.time()),
    )
    if rows and random.random() < 0.02:  # nettoyage occasionnel : une requête de moins par message
        _run("DELETE FROM processed_messages WHERE received_at < ?", (time.time() - 7 * 86400,))
    return bool(rows)


# --- Conversations ------------------------------------------------------------------


@dataclass
class Turn:
    question: str
    answer: str
    question_reformulee: str | None = None


def get_history(user_hash: str) -> list[Turn]:
    """Derniers échanges récents, du plus ancien au plus récent."""
    s = get_settings()
    rows = _query(
        "SELECT question, question_reformulee, answer FROM turns "
        "WHERE user_hash = ? AND created_at >= ? ORDER BY created_at DESC LIMIT ?",
        (user_hash, time.time() - s.history_ttl_minutes * 60, s.history_max_turns),
    )
    return [Turn(r["question"], r["answer"], r["question_reformulee"]) for r in reversed(rows)]


def add_turn(user_hash: str, question: str, answer: str, question_reformulee: str | None = None) -> None:
    _run(
        "INSERT INTO turns(user_hash, created_at, question, question_reformulee, answer) VALUES (?, ?, ?, ?, ?)",
        (user_hash, time.time(), question, question_reformulee, answer),
    )


def clear_history(user_hash: str) -> None:
    _run("DELETE FROM turns WHERE user_hash = ?", (user_hash,))


# --- Préférences ----------------------------------------------------------------------


def get_profil(user_hash: str) -> str | None:
    rows = _query("SELECT profil FROM user_prefs WHERE user_hash = ?", (user_hash,))
    return rows[0]["profil"] if rows else None


def set_profil(user_hash: str, profil: str) -> None:
    now = time.time()
    _run(
        "INSERT INTO user_prefs(user_hash, profil, first_seen, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(user_hash) DO UPDATE SET profil = excluded.profil, updated_at = excluded.updated_at",
        (user_hash, profil, now, now),
    )


# --- Compteurs (quotas) ------------------------------------------------------------------


def _period(kind: str) -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m") if kind == "month" else now.strftime("%Y-%m-%d")


def increment(name: str, kind: str = "day") -> int:
    rows = _query(
        "INSERT INTO counters(name, period, value) VALUES (?, ?, 1) "
        "ON CONFLICT(name, period) DO UPDATE SET value = counters.value + 1 RETURNING value",
        (name, _period(kind)),
    )
    return int(rows[0]["value"])


def get_counter(name: str, kind: str = "day") -> int:
    rows = _query("SELECT value FROM counters WHERE name = ? AND period = ?", (name, _period(kind)))
    return int(rows[0]["value"]) if rows else 0


# --- Suivi d'usage ----------------------------------------------------------------------


def log_interaction(**fields) -> int:
    unknown = set(fields) - _INTERACTION_COLUMNS
    if unknown:
        raise ValueError(f"colonnes inconnues : {unknown}")
    if "exercices" in fields and not isinstance(fields["exercices"], str):
        fields["exercices"] = json.dumps(fields["exercices"])
    fields.setdefault("created_at", time.time())
    columns = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    rows = _query(f"INSERT INTO interactions({columns}) VALUES ({placeholders}) RETURNING id", tuple(fields.values()))
    return int(rows[0]["id"])


def set_feedback(interaction_id: int, value: int, user_hash: str | None = None) -> bool:
    """Enregistre un avis. Avec user_hash, seul l'auteur de la question peut la noter."""
    if user_hash is None:
        return _run("UPDATE interactions SET feedback = ? WHERE id = ?", (value, interaction_id)) > 0
    return (
        _run(
            "UPDATE interactions SET feedback = ? WHERE id = ? AND user_hash = ?",
            (value, interaction_id, user_hash),
        )
        > 0
    )


def usage_stats(days: int = 30) -> dict:
    since = time.time() - days * 86400
    total = _query(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT user_hash) AS u, AVG(latence_ms) AS lat, "
        "SUM(tokens_entree) AS tin, SUM(tokens_sortie) AS tout, "
        "SUM(CASE WHEN erreur IS NOT NULL THEN 1 ELSE 0 END) AS err, "
        "SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) AS up, "
        "SUM(CASE WHEN feedback = 0 THEN 1 ELSE 0 END) AS down "
        "FROM interactions WHERE created_at >= ?",
        (since,),
    )[0]
    day = _db().day_expr
    per_day = _query(
        f"SELECT {day} AS jour, COUNT(*) AS n, COUNT(DISTINCT user_hash) AS u "
        f"FROM interactions WHERE created_at >= ? GROUP BY {day} ORDER BY jour",
        (since,),
    )
    by_canal = _query("SELECT canal, COUNT(*) AS n FROM interactions WHERE created_at >= ? GROUP BY canal", (since,))
    by_intention = _query(
        "SELECT intention, COUNT(*) AS n FROM interactions WHERE created_at >= ? GROUP BY intention", (since,)
    )
    return {
        "periode_jours": days,
        "stockage": backend_name(),
        "questions": int(total["n"] or 0),
        "utilisateurs": int(total["u"] or 0),
        "latence_moyenne_ms": round(float(total["lat"] or 0)),
        "tokens_entree": int(total["tin"] or 0),
        "tokens_sortie": int(total["tout"] or 0),
        "erreurs": int(total["err"] or 0),
        "avis_positifs": int(total["up"] or 0),
        "avis_negatifs": int(total["down"] or 0),
        "reranks_pinecone_ce_mois": get_counter("rerank_pinecone", "month"),
        "par_jour": [{"jour": r["jour"], "n": int(r["n"]), "u": int(r["u"])} for r in per_day],
        "par_canal": {r["canal"]: int(r["n"]) for r in by_canal},
        "par_intention": {r["intention"]: int(r["n"]) for r in by_intention},
    }
