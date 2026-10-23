"""Webhook WhatsApp Cloud API.

GET  /whatsapp/webhook : vérification de l'abonnement par Meta (hub.challenge).
POST /whatsapp/webhook : messages entrants. La signature X-Hub-Signature-256 est
vérifiée. Sur un serveur classique, Meta reçoit un 200 immédiatement et la
réponse est préparée en arrière-plan. En serverless (Vercel), le message est
traité avant la réponse : un renvoi de Meta entre-temps est écarté comme doublon.
"""

import hashlib
import hmac
import json
import logging
from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from rag_fisc_v2 import storage
from rag_fisc_v2.channels.whatsapp.handler import WhatsAppHandler, parse_webhook
from rag_fisc_v2.config import get_settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])


@lru_cache
def get_handler() -> WhatsAppHandler:
    from rag_fisc_v2.api.routes import get_rag_service
    from rag_fisc_v2.channels.whatsapp.client import WhatsAppClient

    return WhatsAppHandler(get_rag_service(), WhatsAppClient())


def verify_signature(body: bytes, header: str | None, app_secret: str) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


@router.get("/webhook", response_class=PlainTextResponse, summary="Vérification du webhook par Meta")
def verify(
    mode: str = Query(alias="hub.mode"),
    token: str = Query(alias="hub.verify_token"),
    challenge: str = Query(alias="hub.challenge"),
) -> str:
    expected = get_settings().whatsapp_verify_token
    if mode == "subscribe" and expected and hmac.compare_digest(token, expected.get_secret_value()):
        return challenge
    raise HTTPException(status_code=403, detail="Jeton de vérification invalide")


@router.post("/webhook", summary="Messages WhatsApp entrants")
async def receive(request: Request, background: BackgroundTasks) -> dict:
    settings = get_settings()
    if settings.whatsapp_app_secret is None:
        raise HTTPException(status_code=503, detail="WHATSAPP_APP_SECRET non configuré")
    body = await request.body()
    if not verify_signature(
        body, request.headers.get("X-Hub-Signature-256"), settings.whatsapp_app_secret.get_secret_value()
    ):
        raise HTTPException(status_code=403, detail="Signature invalide")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="JSON invalide") from exc

    handler = get_handler()
    for msg in parse_webhook(payload):
        # Meta peut renvoyer le même message : on ne le traite qu'une fois. Le message est
        # marqué avant traitement, donc un renvoi pendant un traitement long est ignoré.
        if not storage.mark_message_processed(msg.message_id):
            continue
        if settings.whatsapp_inline:
            # Serverless : traitement avant la réponse, car rien ne s'exécute de façon sûre après.
            await run_in_threadpool(handler.handle, msg)
        else:
            background.add_task(handler.handle, msg)
    return {"status": "ok"}
