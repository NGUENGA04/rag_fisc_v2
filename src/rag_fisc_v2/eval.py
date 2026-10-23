"""Évaluation de la chaîne sur un jeu de questions annotées."""

from pathlib import Path

import typer
import yaml

from rag_fisc_v2.domain import Profil
from rag_fisc_v2.ingestion.manifest import load_manifest
from rag_fisc_v2.retrieval.analysis import analyze_question
from rag_fisc_v2.retrieval.retriever import FiscalRetriever, Passage
from rag_fisc_v2.schemas.api import AskRequest
from rag_fisc_v2.services.rag import RagService


def _articles_hit(passages: list[Passage], expected: list[str]) -> bool:
    for p in passages:
        found = {p.metadata.get("article")} | set(p.metadata.get("articles_cites", []))
        if p.doc_type.value != "cgi":
            found.discard(p.metadata.get("article"))  # numéro propre à la loi ou à la circulaire
        if found & set(expected):
            return True
    return False


def run_eval(path: Path, generate: bool = False) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = load_manifest()
    retriever = FiscalRetriever(manifest)
    service = RagService(manifest, retriever, track=False) if generate else None

    totals = {"questions": 0, "checks": 0, "ok": 0}
    for q in data["questions"]:
        expected = q.get("attendu", {})
        analysis = analyze_question(q["question"])
        if q.get("exercice"):
            analysis.exercices = [q["exercice"]]
        checks: dict[str, bool] = {}

        if "intention" in expected:
            checks["intention"] = analysis.intention == expected["intention"]

        if analysis.intention != "hors_sujet" and ("articles" in expected or "doc_ids" in expected):
            passages = retriever.retrieve(analysis).passages
            if "articles" in expected:
                checks["articles"] = _articles_hit(passages, expected["articles"])
            if "doc_ids" in expected:
                checks["doc_ids"] = any(p.metadata.get("doc_id") in expected["doc_ids"] for p in passages)

        if service and "contient" in expected:
            answer = service.ask(
                AskRequest(question=q["question"], profil=Profil(q.get("profil", "grand_public")), exercice=q.get("exercice"))
            ).reponse
            checks["contient"] = all(s in answer for s in expected["contient"])

        totals["questions"] += 1
        totals["checks"] += len(checks)
        totals["ok"] += sum(checks.values())
        status = " ".join(f"{k}={'ok' if v else 'ÉCHEC'}" for k, v in checks.items()) or "(aucun contrôle)"
        typer.echo(f"{q['id']:26} {status}")

    rate = totals["ok"] / totals["checks"] if totals["checks"] else 0
    typer.echo(f"\n{totals['ok']}/{totals['checks']} contrôles réussis ({rate:.0%}) sur {totals['questions']} questions")
    return totals
