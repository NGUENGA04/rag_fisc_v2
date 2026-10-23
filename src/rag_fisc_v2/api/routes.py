"""Routes HTTP de l'API."""

import hmac
import re
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from rag_fisc_v2 import storage
from rag_fisc_v2.clients import MissingCredentialsError
from rag_fisc_v2.config import get_settings
from rag_fisc_v2.ingestion.baremes import available_baremes, get_bareme
from rag_fisc_v2.ingestion.indexer import load_index_state
from rag_fisc_v2.ingestion.manifest import Manifest, load_manifest
from rag_fisc_v2.schemas.api import (
    AskRequest,
    AskResponse,
    BaremeResponse,
    DocumentInfo,
    FeedbackRequest,
    HealthResponse,
)
from rag_fisc_v2.services.rag import RagService, _to_calcul

router = APIRouter()


@lru_cache
def get_manifest() -> Manifest:
    return load_manifest()


@lru_cache
def get_rag_service() -> RagService:
    return RagService(get_manifest())


@router.post("/ask", response_model=AskResponse, summary="Poser une question fiscale")
def ask(request: AskRequest, service: RagService = Depends(get_rag_service)) -> AskResponse:
    try:
        return service.ask(request)
    except MissingCredentialsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/feedback", status_code=204, summary="Noter une réponse (utile ou non)")
def feedback(request: FeedbackRequest) -> None:
    if not storage.set_feedback(request.interaction_id, int(request.utile)):
        raise HTTPException(status_code=404, detail="Interaction inconnue")


@router.get("/admin/stats", summary="Statistiques d'usage (jeton administrateur requis)")
def admin_stats(
    days: int = Query(default=30, ge=1, le=365),
    x_admin_token: str | None = Header(default=None),
) -> dict:
    expected = get_settings().admin_token
    if expected is None or not expected.get_secret_value():
        raise HTTPException(status_code=404)
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected.get_secret_value()):
        raise HTTPException(status_code=401, detail="Jeton administrateur invalide")
    return storage.usage_stats(days)


@router.get("/bareme/{nom}", response_model=BaremeResponse, summary="Calcul exact à partir d'un barème mensuel")
def bareme(
    nom: Literal["irpp", "tdl"],
    montant: int = Query(gt=0, description="Salaire mensuel en FCFA"),
) -> BaremeResponse:
    table = get_bareme(f"bareme-{nom}")
    if table is None:
        raise HTTPException(status_code=404, detail="Barème non extrait. Lancez : rag-fisc-v2 ingest --doc bareme-" + nom)
    result = table.lookup(montant)
    return BaremeResponse(**_to_calcul(result).model_dump(), texte=result.as_text())


@router.get("/documents", response_model=list[DocumentInfo], summary="Documents du corpus et état d'indexation")
def documents(manifest: Manifest = Depends(get_manifest)) -> list[DocumentInfo]:
    state = load_index_state()
    return [
        DocumentInfo(
            doc_id=d.doc_id,
            titre=d.titre,
            type=d.type,
            exercice=d.exercice,
            reference=d.reference,
            fichier_present=d.path().exists(),
            indexe=d.doc_id in state,
            morceaux=state.get(d.doc_id, {}).get("chunks"),
            indexe_le=state.get(d.doc_id, {}).get("indexed_at"),
        )
        for d in manifest.documents
    ]


def _configured(secret) -> bool:
    return secret is not None and bool(secret.get_secret_value().strip())


@router.get("/health", response_model=HealthResponse, summary="État du service")
def health() -> HealthResponse:
    settings = get_settings()
    state = load_index_state()
    mistral = _configured(settings.mistral_api_key)
    pinecone = _configured(settings.pinecone_api_key)
    modeles = {
        "ocr": settings.ocr_model,
        "embeddings": settings.embed_model,
        "analyse": settings.analysis_model,
        "generation": settings.generation_model,
    }
    # Un nom de modèle ne contient ni "=", ni espace : sinon la variable a été mal recopiée.
    modeles_valides = all(re.fullmatch(r"[a-z0-9][a-z0-9.\-]*", m or "") for m in modeles.values())
    try:
        stockage = storage.backend_name()
    except Exception as exc:  # base injoignable : le signaler sans faire échouer la route
        stockage = f"erreur : {type(exc).__name__}"
    return HealthResponse(
        statut="ok"
        if mistral and pinecone and state and modeles_valides and not stockage.startswith("erreur")
        else "incomplet",
        stockage=stockage,
        modeles=modeles,
        modeles_valides=modeles_valides,
        mistral_configure=mistral,
        pinecone_configure=pinecone,
        whatsapp_configure=bool(
            _configured(settings.whatsapp_access_token)
            and settings.whatsapp_phone_number_id
            and _configured(settings.whatsapp_app_secret)
            and _configured(settings.whatsapp_verify_token)
        ),
        documents_indexes=len(state),
        baremes_disponibles=available_baremes(),
    )
