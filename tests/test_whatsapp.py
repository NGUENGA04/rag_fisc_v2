import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from rag_fisc_v2 import storage
from rag_fisc_v2.channels.whatsapp import webhook
from rag_fisc_v2.channels.whatsapp.formatter import (
    format_response,
    markdown_to_whatsapp,
    renumber_citations,
    split_message,
)
from rag_fisc_v2.channels.whatsapp.handler import (
    EXPERT_ON,
    FEEDBACK_THANKS,
    TEXT_ONLY,
    WELCOME,
    IncomingMessage,
    WhatsAppHandler,
    detect_command,
    parse_webhook,
)
from rag_fisc_v2.config import get_settings
from rag_fisc_v2.domain import Canal, DocType, Profil
from rag_fisc_v2.main import app
from rag_fisc_v2.schemas.api import AskResponse, Source


def _source(ref: str, article: str = "17", doc_type: DocType = DocType.CGI, exercice: int = 2026) -> Source:
    return Source(
        ref=ref, doc_id="cgi-2026", type=doc_type, titre_document="CGI", exercice=exercice,
        article=article, page_debut=21, page_fin=21, extrait="…", score=0.9, citee=True,
    )


# --- Mise en forme -----------------------------------------------------------------


def test_markdown_conversion():
    md = "## Taux\n**Important** : voir le tableau\n| Taux | Valeur |\n|---|---|\n| IS | 30 % |\n* point"
    out = markdown_to_whatsapp(md)
    assert "*Taux*" in out
    assert "*Important*" in out and "**" not in out
    assert "- IS : 30 %" in out
    assert "|---|" not in out
    assert "- point" in out


def test_citations_renumbered_in_order_and_unknown_removed():
    text = "Le taux est de 30 % [S4]. Acomptes [S2][S4]. Barème [B]. Inventé [S9]."
    out, footer = renumber_citations(text, [_source("S2", "21"), _source("S4", "17")], has_calcul=True)
    assert out == "Le taux est de 30 % [1]. Acomptes [2][1]. Barème [3]. Inventé."
    assert footer[0] == "[1] CGI 2026, art. 17, p. 21"
    assert footer[1].startswith("[2] CGI 2026, art. 21")
    assert footer[2].startswith("[3] Barème")


def test_split_message_respects_limit():
    text = "\n\n".join(["Paragraphe numéro un. " * 20] * 6)
    parts = split_message(text, 500)
    assert len(parts) > 1
    assert all(len(p) <= 500 for p in parts)


def test_format_response_adds_sources_and_warnings():
    response = AskResponse(
        reponse="Le taux de l'IS est de 30 % [S1].",
        profil=Profil.GRAND_PUBLIC,
        exercices=[2025],
        sources=[_source("S1")],
        avertissements=["Le CGI de l'exercice 2025 n'est pas dans le corpus.", "Cette réponse est fournie à titre informatif."],
    )
    messages = format_response(response)
    full = "\n".join(messages)
    assert "[1]" in full and "📚 *Sources*" in full
    assert "⚠️ Le CGI de l'exercice 2025" in full
    assert "Cette réponse est fournie" not in full  # remplacée par la mention courte
    assert "Information indicative" in full


# --- Commandes et analyse du webhook ----------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Mode Expert", "expert"),
        ("mode simple !", "simple"),
        ("AIDE", "aide"),
        ("?", "aide"),
        ("Nouveau", "nouveau"),
        ("Bonjour", "salutation"),
        ("Bonjour, quel est le taux de l'IS ?", None),
        ("Quel est le taux de la TVA ?", None),
    ],
)
def test_detect_command(text, expected):
    assert detect_command(text) == expected


def _payload(messages: list[dict]) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "1", "changes": [{"field": "messages", "value": {"messaging_product": "whatsapp", "messages": messages}}]}],
    }


def test_parse_webhook_text_button_and_status():
    payload = _payload(
        [
            {"from": "237600000001", "id": "wamid.1", "type": "text", "text": {"body": "Taux IS ?"}},
            {"from": "237600000001", "id": "wamid.2", "type": "interactive", "interactive": {"type": "button_reply", "button_reply": {"id": "fb:1:5", "title": "Oui"}}},
            {"from": "237600000001", "id": "wamid.3", "type": "audio", "audio": {"id": "x"}},
        ]
    )
    payload["entry"][0]["changes"].append({"field": "messages", "value": {"statuses": [{"id": "wamid.0", "status": "delivered"}]}})
    msgs = parse_webhook(payload)
    assert [m.type for m in msgs] == ["text", "interactive", "audio"]
    assert msgs[0].text == "Taux IS ?"
    assert msgs[1].button_id == "fb:1:5"


# --- Traitement des messages ---------------------------------------------------------------


class FakeClient:
    def __init__(self):
        self.sent: list[str] = []
        self.buttons: list[list] = []

    def send_text(self, to, body):
        self.sent.append(body)

    def send_buttons(self, to, body, buttons):
        self.buttons.append(buttons)

    def mark_read(self, message_id, typing=True):
        pass


class FakeService:
    def __init__(self):
        self.requests = []

    def ask(self, request, canal=Canal.API):
        self.requests.append((request, canal))
        interaction = storage.log_interaction(
            canal=canal.value, user_hash=storage.hash_user(f"{canal.value}:{request.conversation_id}"), question=request.question
        )
        return AskResponse(
            reponse="Le taux est de 30 % [S1].", profil=request.profil, exercices=[2026],
            sources=[_source("S1")], interaction_id=interaction,
        )


def _text(body: str, mid: str = "wamid.x") -> IncomingMessage:
    return IncomingMessage(wa_id="237600000001", message_id=mid, type="text", text=body)


def test_first_question_gets_welcome_then_answer_and_feedback_buttons():
    client, service = FakeClient(), FakeService()
    handler = WhatsAppHandler(service, client)
    handler.handle(_text("Quel est le taux de l'IS ?"))
    assert client.sent[0] == WELCOME
    assert "30 % [1]" in client.sent[1]
    request, canal = service.requests[0]
    assert canal == Canal.WHATSAPP
    assert request.profil == Profil.GRAND_PUBLIC
    assert request.conversation_id == "237600000001"
    assert client.buttons and client.buttons[0][0][0].startswith("fb:1:")


def test_expert_mode_is_remembered():
    client, service = FakeClient(), FakeService()
    handler = WhatsAppHandler(service, client)
    handler.handle(_text("mode expert", "m1"))
    assert client.sent[-1] == EXPERT_ON
    handler.handle(_text("Quel est le taux de l'IS ?", "m2"))
    assert service.requests[0][0].profil == Profil.EXPERT


def test_audio_is_refused_politely():
    client = FakeClient()
    WhatsAppHandler(FakeService(), client).handle(IncomingMessage(wa_id="1", message_id="a", type="audio"))
    assert client.sent[-1] == TEXT_ONLY


def test_daily_limit():
    settings = get_settings()
    previous = settings.whatsapp_daily_limit_per_user
    settings.whatsapp_daily_limit_per_user = 2
    try:
        client, service = FakeClient(), FakeService()
        handler = WhatsAppHandler(service, client)
        for i in range(3):
            handler.handle(_text("Quel est le taux de l'IS ?", f"m{i}"))
        assert len(service.requests) == 2
        assert "limite de 2 questions" in client.sent[-1]
    finally:
        settings.whatsapp_daily_limit_per_user = previous


def test_feedback_button_records_only_for_owner():
    client, service = FakeClient(), FakeService()
    handler = WhatsAppHandler(service, client)
    handler.handle(_text("Quel est le taux de l'IS ?"))
    button_id = client.buttons[0][0][0]  # fb:1:<id>
    handler.handle(IncomingMessage(wa_id="237600000001", message_id="b1", type="interactive", button_id=button_id))
    assert client.sent[-1] == FEEDBACK_THANKS
    # Un autre numéro ne peut pas noter cette réponse.
    other = FakeClient()
    WhatsAppHandler(service, other).handle(
        IncomingMessage(wa_id="237699999999", message_id="b2", type="interactive", button_id=button_id)
    )
    assert FEEDBACK_THANKS not in other.sent
    assert storage.usage_stats()["avis_positifs"] == 1


def test_service_error_sends_apology():
    class Broken(FakeService):
        def ask(self, request, canal=Canal.API):
            raise RuntimeError("panne")

    client = FakeClient()
    WhatsAppHandler(Broken(), client).handle(_text("Quel est le taux de l'IS ?"))
    assert "erreur technique" in client.sent[-1]


# --- Webhook HTTP ---------------------------------------------------------------------------


@pytest.fixture
def wa_settings():
    from pydantic import SecretStr

    s = get_settings()
    saved = (s.whatsapp_app_secret, s.whatsapp_verify_token)
    s.whatsapp_app_secret = SecretStr("secret-test")
    s.whatsapp_verify_token = SecretStr("verif-test")
    yield s
    s.whatsapp_app_secret, s.whatsapp_verify_token = saved


def _sign(body: bytes) -> str:
    return "sha256=" + hmac.new(b"secret-test", body, hashlib.sha256).hexdigest()


def test_webhook_verification(wa_settings):
    client = TestClient(app)
    ok = client.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "verif-test", "hub.challenge": "42"})
    assert ok.status_code == 200 and ok.text == "42"
    bad = client.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "faux", "hub.challenge": "42"})
    assert bad.status_code == 403


def test_webhook_rejects_bad_signature_and_dedupes(wa_settings, monkeypatch):
    handled: list[str] = []

    class Recorder:
        def handle(self, msg):
            handled.append(msg.message_id)

    monkeypatch.setattr(webhook, "get_handler", lambda: Recorder())
    client = TestClient(app)
    body = json.dumps(_payload([{"from": "1", "id": "wamid.dup", "type": "text", "text": {"body": "Taux IS ?"}}])).encode()

    assert client.post("/whatsapp/webhook", content=body, headers={"X-Hub-Signature-256": "sha256=faux"}).status_code == 403
    for _ in range(2):  # Meta renvoie le même message
        r = client.post("/whatsapp/webhook", content=body, headers={"X-Hub-Signature-256": _sign(body), "Content-Type": "application/json"})
        assert r.status_code == 200
    assert handled == ["wamid.dup"]


def test_inline_mode_processes_before_responding(wa_settings, monkeypatch):
    """En serverless, le message est traité pendant la requête, et un renvoi est ignoré."""
    handled: list[str] = []

    class Recorder:
        def handle(self, msg):
            handled.append(msg.message_id)

    monkeypatch.setattr(wa_settings, "whatsapp_inline_processing", True)
    monkeypatch.setattr(webhook, "get_handler", lambda: Recorder())
    client = TestClient(app)
    body = json.dumps(_payload([{"from": "1", "id": "wamid.inline", "type": "text", "text": {"body": "Taux IS ?"}}])).encode()
    headers = {"X-Hub-Signature-256": _sign(body), "Content-Type": "application/json"}
    assert client.post("/whatsapp/webhook", content=body, headers=headers).status_code == 200
    assert handled == ["wamid.inline"]
    client.post("/whatsapp/webhook", content=body, headers=headers)
    assert handled == ["wamid.inline"]


def test_inline_mode_auto_on_vercel(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "whatsapp_inline_processing", None)
    monkeypatch.delenv("VERCEL", raising=False)
    assert s.whatsapp_inline is False
    monkeypatch.setenv("VERCEL", "1")
    assert s.whatsapp_inline is True


def test_delete_my_data_command():
    client, service = FakeClient(), FakeService()
    handler = WhatsAppHandler(service, client)
    handler.handle(_text("Quel est le taux de l'IS ?", "d1"))
    user_hash = storage.hash_user("whatsapp:237600000001")
    assert storage.get_profil(user_hash) is not None
    assert storage.usage_stats()["questions"] == 1
    handler.handle(_text("Supprimer mes données", "d2"))
    assert "supprimées" in client.sent[-1]
    assert storage.get_profil(user_hash) is None
    assert storage.usage_stats()["questions"] == 0


def test_old_turns_are_purged():
    import time

    storage.add_turn("u", "q", "r")
    storage._run("UPDATE turns SET created_at = ?", (time.time() - 2 * 86400,))
    assert storage.purge_old_turns() == 1
