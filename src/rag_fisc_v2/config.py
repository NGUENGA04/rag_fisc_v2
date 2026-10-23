"""Configuration centrale, lue depuis les variables d'environnement et le fichier .env."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Clés d'API -------------------------------------------------------
    mistral_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None

    # --- Modèles Mistral ----------------------------------------------------
    # Les alias "-latest" suivent la dernière version. Pour figer une version
    # (par exemple Small 4 ou Medium 3.5), mettre son identifiant exact dans .env.
    ocr_model: str = "mistral-ocr-latest"
    embed_model: str = "mistral-embed"
    embed_dimension: int = 1024
    analysis_model: str = "mistral-small-latest"
    generation_model: str = "mistral-medium-latest"
    generation_max_tokens: int = 2000

    # --- Pinecone -----------------------------------------------------------
    pinecone_index_name: str = "rag-fisc-cm"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "cgi-cm"
    rerank_model: str = "bge-reranker-v2-m3"

    # --- Chemins ------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    # Base applicative : conversations, suivi d'usage, compteurs.
    # DATABASE_URL défini (Neon, Postgres) : PostgreSQL. Sinon : SQLite à app_db_path,
    # à placer sur un volume persistant en production.
    database_url: SecretStr | None = None
    app_db_path: Path = PROJECT_ROOT / "data" / "app.db"

    # --- Ingestion ----------------------------------------------------------
    ocr_pages_per_part: int = 20
    chunk_max_chars: int = 3500
    chunk_min_chars: int = 300
    embed_batch_size: int = 32

    # --- Recherche ----------------------------------------------------------
    top_k_dense: int = 20
    top_k_article: int = 8
    top_k_linked: int = 8
    rerank_top_n: int = 8
    rerank_min_score: float = 0.02
    # auto : Pinecone tant que le budget mensuel n'est pas atteint, puis Mistral Small.
    rerank_provider: Literal["auto", "pinecone", "mistral", "none"] = "auto"
    # L'offre gratuite de Pinecone inclut 500 reranks par mois : on garde une marge.
    pinecone_rerank_monthly_budget: int = 450
    mistral_rerank_min_score: float = 0.3

    # --- Conversations ------------------------------------------------------
    history_max_turns: int = 4
    history_ttl_minutes: int = 120
    # Sel du hachage des numéros de téléphone : aucun numéro n'est stocké en clair.
    user_hash_salt: SecretStr = SecretStr("changer-ce-sel-en-production")

    # --- WhatsApp (API Cloud de Meta) ---------------------------------------
    whatsapp_access_token: SecretStr | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_app_secret: SecretStr | None = None
    whatsapp_verify_token: SecretStr | None = None
    whatsapp_graph_version: str = "v23.0"
    whatsapp_daily_limit_per_user: int = 30
    whatsapp_max_message_chars: int = 3500
    # Traitement du message dans la requête du webhook plutôt qu'en arrière-plan.
    # Indispensable en serverless (Vercel), où rien ne garantit l'exécution après la réponse.
    # Non défini : activé automatiquement si la variable VERCEL est présente.
    whatsapp_inline_processing: bool | None = None

    @property
    def whatsapp_inline(self) -> bool:
        if self.whatsapp_inline_processing is not None:
            return self.whatsapp_inline_processing
        return bool(os.environ.get("VERCEL"))

    # --- Administration -----------------------------------------------------
    # Jeton requis pour /admin/stats. Sans jeton configuré, la route est désactivée.
    admin_token: SecretStr | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def ocr_dir(self) -> Path:
        return self.data_dir / "ocr"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def baremes_dir(self) -> Path:
        return self.data_dir / "baremes"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.yaml"

    @property
    def index_state_path(self) -> Path:
        return self.data_dir / "index_state.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
