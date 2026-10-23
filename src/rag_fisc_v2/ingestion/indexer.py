"""Vectorisation des morceaux et écriture idempotente dans Pinecone.

Chaque vecteur a pour identifiant "<doc_id>#<numéro>". Réindexer un document
supprime d'abord tous ses vecteurs par préfixe, puis réinsère. Un fichier
d'état (data/index_state.json) mémorise l'empreinte des morceaux indexés :
un document inchangé n'est ni revectorisé ni refacturé.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime

from llama_index.core.schema import MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from rag_fisc_v2.clients import get_embed_model, get_pinecone_index
from rag_fisc_v2.config import get_settings
from rag_fisc_v2.domain import Chunk

log = logging.getLogger(__name__)

# Métadonnées incluses dans le texte vectorisé : elles situent le passage.
EMBED_METADATA_KEYS = ("titre_document", "hierarchie", "article")
# Libellés utilisés dans le texte vectorisé.
_EMBED_LABELS = {"titre_document": "Document", "hierarchie": "Plan", "article": "Article"}


def chunk_to_node(chunk: Chunk) -> TextNode:
    node = TextNode(
        id_=chunk.chunk_id,
        text=chunk.text,
        metadata=dict(chunk.metadata),
        excluded_embed_metadata_keys=[k for k in chunk.metadata if k not in EMBED_METADATA_KEYS],
        excluded_llm_metadata_keys=list(chunk.metadata),
        metadata_template="{key}: {value}",
        text_template="{metadata_str}\n\n{content}",
    )
    # Rattachement au document source : LlamaIndex en déduit le champ doc_id.
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=chunk.metadata["doc_id"])
    return node


def node_to_pinecone_record(node: TextNode) -> dict:
    """Enregistrement Pinecone au format attendu par le PineconeVectorStore de LlamaIndex.

    L'écriture est faite ici plutôt que par PineconeVectorStore.add, qui préfixe
    l'identifiant par le document source ("cgi-2026#cgi-2026#00001").
    """
    metadata = node_to_metadata_dict(node, remove_text=False, flat_metadata=False)
    return {"id": node.node_id, "values": node.embedding, "metadata": metadata}


def embed_text(node: TextNode) -> str:
    """Texte réellement envoyé au modèle d'embedding, avec libellés français."""
    header = "\n".join(
        f"{_EMBED_LABELS[k]} : {node.metadata[k]}" for k in EMBED_METADATA_KEYS if node.metadata.get(k)
    )
    return f"{header}\n\n{node.get_content(MetadataMode.NONE)}" if header else node.get_content(MetadataMode.NONE)


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text.encode())
        h.update(json.dumps(c.metadata, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def load_index_state() -> dict:
    path = get_settings().index_state_path
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_index_state(state: dict) -> None:
    path = get_settings().index_state_path
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_document_vectors(doc_id: str) -> int:
    """Supprime tous les vecteurs d'un document. Fonctionne sur un index serverless."""
    settings = get_settings()
    index = get_pinecone_index()
    deleted = 0
    for page in index.list(prefix=f"{doc_id}#", namespace=settings.pinecone_namespace):
        ids = [v.id for v in page.vectors]
        if ids:
            index.delete(ids=ids, namespace=settings.pinecone_namespace)
            deleted += len(ids)
    return deleted


def index_chunks(doc_id: str, chunks: list[Chunk], force: bool = False) -> int:
    """Indexe les morceaux d'un document. Retourne le nombre de vecteurs écrits."""
    settings = get_settings()
    state = load_index_state()
    fingerprint = chunks_fingerprint(chunks)

    if not force and state.get(doc_id, {}).get("fingerprint") == fingerprint:
        log.info("%s : déjà indexé, rien à faire", doc_id)
        return 0

    removed = delete_document_vectors(doc_id)
    if removed:
        log.info("%s : %d anciens vecteurs supprimés", doc_id, removed)

    nodes = [chunk_to_node(c) for c in chunks]
    embed_model = get_embed_model()
    texts = [embed_text(n) for n in nodes]
    batch = settings.embed_batch_size
    for start in range(0, len(nodes), batch):
        vectors = embed_model.get_text_embedding_batch(texts[start : start + batch])
        for node, vector in zip(nodes[start : start + batch], vectors, strict=True):
            node.embedding = vector
        log.info("%s : %d/%d morceaux vectorisés", doc_id, min(start + batch, len(nodes)), len(nodes))

    get_pinecone_index().upsert(
        vectors=[node_to_pinecone_record(n) for n in nodes],
        namespace=settings.pinecone_namespace,
        batch_size=100,
        show_progress=False,
    )

    state[doc_id] = {
        "fingerprint": fingerprint,
        "chunks": len(chunks),
        "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save_index_state(state)
    return len(nodes)
