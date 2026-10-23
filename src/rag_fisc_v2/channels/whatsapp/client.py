"""Client minimal de l'API WhatsApp Cloud de Meta (Graph API)."""

import logging
import time

import httpx

from rag_fisc_v2.clients import MissingCredentialsError
from rag_fisc_v2.config import get_settings

log = logging.getLogger(__name__)

# Erreurs passagères de la Graph API : limite de débit et indisponibilité du serveur.
_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3


class WhatsAppClient:
    def __init__(self, http: httpx.Client | None = None):
        s = get_settings()
        if not (s.whatsapp_access_token and s.whatsapp_phone_number_id):
            raise MissingCredentialsError(
                "WHATSAPP_ACCESS_TOKEN et WHATSAPP_PHONE_NUMBER_ID doivent être définis dans .env."
            )
        self.url = f"https://graph.facebook.com/{s.whatsapp_graph_version}/{s.whatsapp_phone_number_id}/messages"
        self.http = http or httpx.Client(
            timeout=20,
            headers={"Authorization": f"Bearer {s.whatsapp_access_token.get_secret_value()}"},
        )

    def _post(self, payload: dict) -> dict:
        payload = {"messaging_product": "whatsapp", **payload}
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self.http.post(self.url, json=payload)
            except httpx.TransportError as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise
                log.warning("WhatsApp : erreur réseau (%s), nouvelle tentative", exc)
                time.sleep(attempt)
                continue
            if response.status_code in _RETRYABLE and attempt < _MAX_ATTEMPTS:
                log.warning("WhatsApp %s, nouvelle tentative", response.status_code)
                time.sleep(attempt)
                continue
            if response.status_code >= 400:
                log.error("WhatsApp %s : %s", response.status_code, response.text[:500])
                response.raise_for_status()
            return response.json()
        raise RuntimeError("inaccessible")

    def send_text(self, to: str, body: str) -> dict:
        return self._post(
            {
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {"preview_url": False, "body": body},
            }
        )

    def send_buttons(self, to: str, body: str, buttons: list[tuple[str, str]]) -> dict:
        """buttons : (identifiant, libellé), au plus 3, libellé de 20 caractères maximum."""
        return self._post(
            {
                "recipient_type": "individual",
                "to": to,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {"text": body},
                    "action": {
                        "buttons": [{"type": "reply", "reply": {"id": i, "title": t[:20]}} for i, t in buttons[:3]]
                    },
                },
            }
        )

    def mark_read(self, message_id: str, typing: bool = True) -> None:
        """Accusé de lecture et indicateur « en train d'écrire ». Non bloquant en cas d'échec."""
        payload: dict = {"status": "read", "message_id": message_id}
        if typing:
            payload["typing_indicator"] = {"type": "text"}
        try:
            self._post(payload)
        except Exception as exc:
            log.info("Accusé de lecture non envoyé : %s", exc)
