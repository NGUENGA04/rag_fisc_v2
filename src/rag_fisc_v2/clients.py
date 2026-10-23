"""Création paresseuse des clients externes (Mistral, Pinecone).

Les clés ne sont exigées qu'au moment où un client est réellement utilisé.
Le reste du projet (découpage, barèmes, tests) fonctionne sans clé.
"""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from rag_fisc_v2.config import get_settings

if TYPE_CHECKING:
    from mistralai.client import Mistral
    from pinecone import Pinecone


log = logging.getLogger(__name__)


class MissingCredentialsError(RuntimeError):
    """Une clé d'API nécessaire n'est pas configurée."""


def _require(value, name: str) -> str:
    if value is None or not value.get_secret_value().strip():
        raise MissingCredentialsError(
            f"{name} est absente. Ajoutez-la dans le fichier .env à la racine du projet."
        )
    return value.get_secret_value()


@lru_cache
def get_mistral() -> "Mistral":
    from mistralai.client import Mistral
    from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

    settings = get_settings()
    # Reprises sur 429, 5xx et coupures réseau, dans la limite de 30 s : compatible avec
    # les 60 s d'une requête Vercel, qui comprend aussi la recherche et l'envoi WhatsApp.
    retry = RetryConfig(
        "backoff",
        BackoffStrategy(initial_interval=1000, max_interval=8000, exponent=2.0, max_elapsed_time=30000),
        retry_connection_errors=True,
    )
    return Mistral(
        api_key=_require(settings.mistral_api_key, "MISTRAL_API_KEY"),
        retry_config=retry,
        timeout_ms=120_000,
    )


@lru_cache
def get_pinecone() -> "Pinecone":
    from pinecone import Pinecone

    settings = get_settings()
    return Pinecone(api_key=_require(settings.pinecone_api_key, "PINECONE_API_KEY"))


# Seules métadonnées utilisées dans les filtres de recherche. Les autres, dont le texte
# complet du passage stocké par LlamaIndex (_node_content), ne sont pas indexées :
# moins de mémoire côté Pinecone, et des filtres plus rapides.
FILTERABLE_FIELDS = ("doc_id", "type", "exercice", "article", "articles_cites", "juridiction")


def ensure_pinecone_index():
    """Crée l'index et son namespace s'ils n'existent pas, puis retourne l'index."""
    from pinecone import ServerlessSpec

    settings = get_settings()
    pc = get_pinecone()
    if not pc.has_index(settings.pinecone_index_name):
        log.info("Création de l'index Pinecone %s", settings.pinecone_index_name)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embed_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            deletion_protection="enabled",  # évite une suppression accidentelle depuis la console
            tags={"projet": "rag-fisc-v2"},
        )
    index = pc.Index(settings.pinecone_index_name)
    _ensure_namespace(index, settings.pinecone_namespace)
    return index


def _ensure_namespace(index, namespace: str) -> None:
    """Crée le namespace avec indexation sélective des métadonnées.

    Si l'offre ou l'API ne le permet pas, le namespace sera créé implicitement à la
    première écriture, avec toutes les métadonnées indexées : moins efficace, mais correct.
    """
    from pinecone.errors import NotFoundError

    try:
        index.describe_namespace(name=namespace)
        return
    except NotFoundError:
        pass
    try:
        index.create_namespace(
            name=namespace,
            schema={"fields": {f: {"filterable": True} for f in FILTERABLE_FIELDS}},
        )
        log.info("Namespace %s créé, métadonnées filtrables : %s", namespace, ", ".join(FILTERABLE_FIELDS))
    except Exception as exc:
        log.warning("Namespace %s non créé avec schéma (%s) : indexation de toutes les métadonnées", namespace, exc)


@lru_cache
def get_pinecone_index():
    return ensure_pinecone_index()


@lru_cache
def get_embed_model():
    from llama_index.embeddings.mistralai import MistralAIEmbedding

    settings = get_settings()
    return MistralAIEmbedding(
        model_name=settings.embed_model,
        api_key=_require(settings.mistral_api_key, "MISTRAL_API_KEY"),
        embed_batch_size=settings.embed_batch_size,
    )


@lru_cache
def get_vector_store():
    from llama_index.vector_stores.pinecone import PineconeVectorStore

    settings = get_settings()
    return PineconeVectorStore(
        pinecone_index=get_pinecone_index(),
        namespace=settings.pinecone_namespace,
    )
