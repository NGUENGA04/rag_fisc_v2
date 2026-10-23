"""Application FastAPI."""

import logging

from fastapi import FastAPI

from rag_fisc_v2.api.routes import router
from rag_fisc_v2.channels.whatsapp.webhook import router as whatsapp_router
from rag_fisc_v2.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")

app = FastAPI(
    title="RAG fiscal Cameroun",
    description=(
        "Questions-réponses sourcées sur la fiscalité camerounaise : Code Général des Impôts, "
        "lois de finances, circulaires et arrêtés de la DGI, barèmes mensuels."
    ),
    version="0.1.0",
)
app.include_router(router)
app.include_router(whatsapp_router)

if get_settings().user_hash_salt.get_secret_value() == "changer-ce-sel-en-production":
    logging.getLogger(__name__).warning(
        "USER_HASH_SALT a sa valeur par défaut : définissez une valeur secrète avant la mise en production."
    )
