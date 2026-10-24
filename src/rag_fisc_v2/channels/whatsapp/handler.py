"""Traitement d'un message WhatsApp entrant : commandes, quota, question, réponse."""

import logging
import re
import unicodedata
from dataclasses import dataclass

from rag_fisc_v2 import storage
from rag_fisc_v2.channels.whatsapp.formatter import format_response
from rag_fisc_v2.config import get_settings
from rag_fisc_v2.domain import Canal, Profil
from rag_fisc_v2.schemas.api import AskRequest

log = logging.getLogger(__name__)

WELCOME = (
    "Bonjour 👋 Je suis un assistant sur la *fiscalité camerounaise*. Je réponds à partir du Code Général "
    "des Impôts, des lois de finances et des circulaires de la DGI, en citant mes sources.\n\n"
    "Exemples :\n"
    "- Quel est le taux de l'impôt sur les sociétés ?\n"
    "- Combien d'IRPP sur un salaire de 350 000 FCFA ?\n"
    "- Qu'est-ce qui a changé pour la TVA en 2026 ?\n\n"
    "Commandes : *mode expert*, *mode simple*, *nouveau* (oublier la conversation), *aide*.\n\n"
    "_Mes réponses sont indicatives et ne remplacent pas l'avis d'un professionnel ou de l'administration fiscale._"
)
HELP = (
    "Posez votre question en une phrase, en précisant si possible l'impôt et l'année.\n\n"
    "*mode expert* : réponses techniques avec références précises\n"
    "*mode simple* : réponses courtes et expliquées\n"
    "*nouveau* : commencer une nouvelle conversation\n\n"
    "Pour calculer l'IRPP ou la TDL, indiquez le salaire mensuel, par exemple : « IRPP sur 450 000 FCFA »."
)
EXPERT_ON = "Mode *expert* activé : réponses techniques, avec articles et conditions. Écrivez *mode simple* pour revenir."
SIMPLE_ON = "Mode *simple* activé : réponses courtes et expliquées. Écrivez *mode expert* pour des réponses techniques."
RESET_DONE = "C'est noté, nous repartons de zéro. Quelle est votre question ?"
DATA_DELETED = (
    "Vos données ont été supprimées : historique de conversation, préférences, questions enregistrées et compteurs. "
    "Si vous écrivez de nouveau, une nouvelle conversation commencera."
)
TEXT_ONLY = "Je ne traite pour l'instant que les messages écrits. Pouvez-vous écrire votre question ?"
LIMIT_REACHED = (
    "Vous avez atteint la limite de {limit} questions pour aujourd'hui. Revenez demain, ou contactez le centre "
    "d'appel de la DGI au 8200."
)
ERROR = "Désolé, une erreur technique m'empêche de répondre pour le moment. Réessayez dans quelques minutes."
FEEDBACK_PROMPT = "Cette réponse vous a-t-elle été utile ?"
FEEDBACK_THANKS = "Merci pour votre retour 🙏"
TOO_SHORT = "Pouvez-vous préciser votre question ? Écrivez *aide* pour voir des exemples."

_GREETINGS = {"bonjour", "bonsoir", "salut", "hello", "hi", "bjr", "slt", "coucou", "allo", "hey"}


def _normalize(text: str) -> str:
    """Minuscules, sans accents ni ponctuation, espaces simples : "Mode Expert !" -> "mode expert"."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def detect_command(text: str) -> str | None:
    t = _normalize(text)
    if t in {"aide", "help", "menu", "commandes", "comment ca marche"} or text.strip() == "?":
        return "aide"
    if t in {"mode expert", "expert", "mode pro", "mode professionnel"}:
        return "expert"
    if t in {"mode simple", "simple", "mode grand public", "grand public", "mode normal"}:
        return "simple"
    if t in {"nouveau", "nouvelle question", "nouvelle conversation", "reset", "recommencer", "effacer"}:
        return "nouveau"
    if t in {"supprimer mes donnees", "supprimer mes données", "effacer mes donnees", "supprimer donnees", "stop donnees"}:
        return "supprimer"
    words = t.split()
    if words and len(words) <= 3 and words[0] in _GREETINGS:
        return "salutation"
    return None


@dataclass
class IncomingMessage:
    wa_id: str  # numéro de l'expéditeur, utilisé uniquement pour répondre
    message_id: str
    type: str
    text: str | None = None
    button_id: str | None = None


def parse_webhook(payload: dict) -> list[IncomingMessage]:
    """Extrait les messages d'une notification Meta. Ignore les accusés de livraison."""
    messages: list[IncomingMessage] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") != "messages":
                continue
            for m in change.get("value", {}).get("messages", []) or []:
                msg = IncomingMessage(wa_id=m.get("from", ""), message_id=m.get("id", ""), type=m.get("type", ""))
                if msg.type == "text":
                    msg.text = (m.get("text") or {}).get("body", "")
                elif msg.type == "interactive":
                    reply = (m.get("interactive") or {}).get("button_reply") or {}
                    msg.button_id = reply.get("id")
                elif msg.type == "button":  # réponse à un modèle de message
                    msg.text = (m.get("button") or {}).get("text")
                if msg.wa_id and msg.message_id:
                    messages.append(msg)
    return messages


class WhatsAppHandler:
    def __init__(self, service, client):
        self.service = service
        self.client = client
        self.settings = get_settings()

    def _reply(self, to: str, text: str) -> None:
        self.client.send_text(to, text)

    def handle(self, msg: IncomingMessage) -> None:
        """Traite un message. Appelé en arrière-plan, après l'accusé de réception à Meta."""
        user_hash = storage.hash_user(f"{Canal.WHATSAPP.value}:{msg.wa_id}")
        try:
            self._handle(msg, user_hash)
        except Exception:
            log.exception("Échec du traitement du message WhatsApp %s", msg.message_id)
            try:
                self._reply(msg.wa_id, ERROR)
            except Exception:
                log.exception("Impossible d'envoyer le message d'erreur")

    def _handle(self, msg: IncomingMessage, user_hash: str) -> None:
        self.client.mark_read(msg.message_id, typing=msg.type == "text")
        first_contact = storage.get_profil(user_hash) is None
        if first_contact:
            storage.set_profil(user_hash, Profil.GRAND_PUBLIC.value)

        if msg.type == "interactive" and msg.button_id:
            self._handle_feedback(msg, user_hash)
            return
        if not msg.text:
            self._reply(msg.wa_id, TEXT_ONLY)
            return

        command = detect_command(msg.text)
        if command == "salutation" or (first_contact and command is None and len(msg.text.strip()) < 3):
            self._reply(msg.wa_id, WELCOME)
            return
        if command == "aide":
            self._reply(msg.wa_id, HELP)
            return
        if command == "expert":
            storage.set_profil(user_hash, Profil.EXPERT.value)
            self._reply(msg.wa_id, EXPERT_ON)
            return
        if command == "simple":
            storage.set_profil(user_hash, Profil.GRAND_PUBLIC.value)
            self._reply(msg.wa_id, SIMPLE_ON)
            return
        if command == "nouveau":
            storage.clear_history(user_hash)
            self._reply(msg.wa_id, RESET_DONE)
            return
        if command == "supprimer":
            storage.delete_user_data(user_hash)
            self._reply(msg.wa_id, DATA_DELETED)
            return

        question = msg.text.strip()[:2000]
        if len(question) < 3:
            self._reply(msg.wa_id, TOO_SHORT)
            return

        limit = self.settings.whatsapp_daily_limit_per_user
        if storage.increment(f"wa_questions:{user_hash}", "day") > limit:
            self._reply(msg.wa_id, LIMIT_REACHED.format(limit=limit))
            return

        if first_contact:
            self._reply(msg.wa_id, WELCOME)

        profil = Profil(storage.get_profil(user_hash) or Profil.GRAND_PUBLIC.value)
        response = self.service.ask(
            AskRequest(question=question, profil=profil, conversation_id=msg.wa_id),
            canal=Canal.WHATSAPP,
        )
        for part in format_response(response, self.settings.whatsapp_max_message_chars):
            self._reply(msg.wa_id, part)

        if response.interaction_id and response.sources:
            self.client.send_buttons(
                msg.wa_id,
                FEEDBACK_PROMPT,
                [(f"fb:1:{response.interaction_id}", "👍 Oui"), (f"fb:0:{response.interaction_id}", "👎 Non")],
            )

    def _handle_feedback(self, msg: IncomingMessage, user_hash: str) -> None:
        m = re.fullmatch(r"fb:([01]):(\d+)", msg.button_id or "")
        if m and storage.set_feedback(int(m.group(2)), int(m.group(1)), user_hash=user_hash):
            self._reply(msg.wa_id, FEEDBACK_THANKS)
