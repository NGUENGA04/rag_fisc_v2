"""Tests de l'API sans appel externe : le service RAG est remplacé par un faux."""

from fastapi.testclient import TestClient

from rag_fisc_v2.api.routes import get_rag_service
from rag_fisc_v2.domain import Profil
from rag_fisc_v2.main import app
from rag_fisc_v2.schemas.api import AskRequest, AskResponse


class FakeService:
    def ask(self, request: AskRequest) -> AskResponse:
        return AskResponse(reponse=f"écho : {request.question}", profil=request.profil, exercices=[2026], sources=[])


app.dependency_overrides[get_rag_service] = lambda: FakeService()
client = TestClient(app)


def test_ask():
    r = client.post("/ask", json={"question": "Quel est le taux de l'IS ?", "profil": "expert"})
    assert r.status_code == 200
    body = r.json()
    assert body["reponse"].startswith("écho")
    assert body["profil"] == Profil.EXPERT


def test_ask_validation():
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "Bonjour test", "profil": "inconnu"}).status_code == 422


def test_documents_lists_manifest():
    r = client.get("/documents")
    assert r.status_code == 200
    ids = {d["doc_id"] for d in r.json()}
    assert {"cgi-2026", "lfi-2025", "bareme-irpp"} <= ids


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "mistral_configure" in r.json()


def test_bareme_route_validates_name():
    assert client.get("/bareme/tva?montant=100000").status_code == 422
