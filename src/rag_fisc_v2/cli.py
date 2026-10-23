"""Ligne de commande : ingestion, inspection, questions et évaluation.

Exemples :
    rag-fisc-v2 docs
    rag-fisc-v2 ingest --doc fiche-is --no-index
    rag-fisc-v2 ingest
    rag-fisc-v2 ask "Quel est le taux de l'IS ?" --profil expert
    rag-fisc-v2 bareme irpp 350000
    rag-fisc-v2 eval
    rag-fisc-v2 serve
"""

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from rag_fisc_v2.config import PROJECT_ROOT, get_settings
from rag_fisc_v2.domain import Canal, Profil

app = typer.Typer(help="RAG fiscal Cameroun", no_args_is_help=True, pretty_exceptions_show_locals=False)

DocOption = Annotated[list[str] | None, typer.Option("--doc", "-d", help="doc_id à traiter (répétable). Tous par défaut.")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "pdfminer", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _setup_logging(verbose)


@contextmanager
def _credentials_required():
    """Affiche un message clair au lieu d'une trace d'erreur quand une clé manque."""
    from rag_fisc_v2.clients import MissingCredentialsError

    try:
        yield
    except MissingCredentialsError as exc:
        typer.secho(f"Configuration incomplète : {exc}", fg=typer.colors.RED, err=True)
        typer.echo("Modèle de fichier : .env.example", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def docs() -> None:
    """Liste les documents du manifeste et leur état."""
    from rag_fisc_v2.domain import DocType
    from rag_fisc_v2.ingestion.baremes import bareme_path
    from rag_fisc_v2.ingestion.indexer import load_index_state
    from rag_fisc_v2.ingestion.manifest import load_manifest

    manifest = load_manifest()
    state = load_index_state()
    for d in manifest.documents:
        present = "ok " if d.path().exists() else "ABSENT"
        if d.type == DocType.BAREME:
            indexed = "table extraite" if bareme_path(d.doc_id).exists() else "table à extraire"
        elif d.doc_id in state:
            indexed = f"{state[d.doc_id]['chunks']} morceaux indexés"
        else:
            indexed = "non indexé"
        typer.echo(f"{d.doc_id:28} {d.type.value:10} {str(d.exercice or '-'):5} {present:6} {indexed}")


@app.command()
def ingest(
    doc: DocOption = None,
    index: Annotated[bool, typer.Option(help="Écrire dans Pinecone. --no-index : OCR et découpage seulement.")] = True,
    force_ocr: Annotated[bool, typer.Option(help="Refaire l'OCR même si le cache est valide.")] = False,
    force_index: Annotated[bool, typer.Option(help="Réindexer même si les morceaux n'ont pas changé.")] = False,
) -> None:
    """OCR, découpage et indexation des documents."""
    from rag_fisc_v2.ingestion.manifest import load_manifest
    from rag_fisc_v2.ingestion.pipeline import ingest as run

    with _credentials_required():
        reports = run(load_manifest(), doc, index=index, force_ocr=force_ocr, force_index=force_index)
    typer.echo("\nRésumé :")
    for r in reports:
        typer.echo(f"  {r.doc_id:28} pages={r.pages:5} morceaux={r.chunks:5} indexés={r.indexed:5} {r.note}")


@app.command()
def chunks(
    doc_id: str,
    article: Annotated[str | None, typer.Option(help="Affiche seulement cet article.")] = None,
    limit: int = 5,
) -> None:
    """Affiche les morceaux produits pour un document, pour relecture."""
    from rag_fisc_v2.ingestion.pipeline import load_chunks

    items = load_chunks(doc_id)
    if article:
        items = [c for c in items if c.metadata.get("article") == article]
    sizes = [len(c.text) for c in load_chunks(doc_id)]
    average = sum(sizes) // max(len(sizes), 1)
    typer.echo(f"{len(sizes)} morceaux, taille moyenne {average} caractères, max {max(sizes, default=0)}")
    for c in items[:limit]:
        meta = {k: v for k, v in c.metadata.items() if k not in ("titre_document", "juridiction", "reference")}
        typer.echo(f"\n=== {c.chunk_id} {json.dumps(meta, ensure_ascii=False)}\n{c.text[:1500]}")


@app.command()
def bareme(nom: str, montant: int) -> None:
    """Calcul exact à partir d'un barème mensuel (irpp ou tdl)."""
    from rag_fisc_v2.ingestion.baremes import get_bareme

    table = get_bareme(f"bareme-{nom}")
    if table is None:
        raise typer.BadParameter(f"Barème absent. Lancez : rag-fisc-v2 ingest --doc bareme-{nom}")
    typer.echo(table.lookup(montant).as_text())


@app.command()
def analyze(question: str, llm: bool = True) -> None:
    """Montre l'analyse d'une question (intention, exercices, articles)."""
    from rag_fisc_v2.retrieval.analysis import analyze_question

    typer.echo(analyze_question(question, use_llm=llm).model_dump_json(indent=2))


@app.command()
def ask(
    question: str,
    profil: Annotated[Profil, typer.Option()] = Profil.GRAND_PUBLIC,
    exercice: Annotated[int | None, typer.Option()] = None,
    debug: bool = False,
) -> None:
    """Pose une question et affiche la réponse avec ses sources."""
    from rag_fisc_v2.ingestion.manifest import load_manifest
    from rag_fisc_v2.schemas.api import AskRequest
    from rag_fisc_v2.services.rag import RagService

    with _credentials_required():
        response = RagService(load_manifest(), track=False).ask(
            AskRequest(question=question, profil=profil, exercice=exercice, debug=debug), canal=Canal.CLI
        )
    typer.echo(f"\n{response.reponse}\n")
    typer.echo(f"Exercice(s) : {', '.join(map(str, response.exercices))}")
    for s in response.sources:
        mark = "*" if s.citee else " "
        typer.echo(f" {mark}[{s.ref}] {s.titre_document} | art. {s.article or '-'} | p. {s.page_debut} | score {s.score}")
    for w in response.avertissements:
        typer.echo(f"  ! {w}")
    if debug and response.analyse:
        typer.echo(json.dumps(response.analyse, ensure_ascii=False, indent=2))


@app.command(name="eval")
def evaluate(
    path: Annotated[Path, typer.Option()] = PROJECT_ROOT / "eval" / "questions.yaml",
    generate: Annotated[bool, typer.Option(help="Génère aussi les réponses (plus lent et payant).")] = False,
) -> None:
    """Évalue la recherche sur le jeu de questions : l'article attendu est-il retrouvé ?"""
    from rag_fisc_v2.eval import run_eval

    with _credentials_required():
        run_eval(path, generate=generate)


@app.command()
def stats(days: int = 30) -> None:
    """Statistiques d'usage : questions, utilisateurs, avis, coûts."""
    from rag_fisc_v2.storage import usage_stats

    s = usage_stats(days)
    typer.echo(f"Sur {days} jours : {s['questions']} questions, {s['utilisateurs']} utilisateurs distincts")
    typer.echo(f"Latence moyenne : {s['latence_moyenne_ms']} ms, erreurs : {s['erreurs']}")
    typer.echo(f"Avis : {s['avis_positifs']} utiles, {s['avis_negatifs']} pas utiles")
    typer.echo(f"Jetons Mistral : {s['tokens_entree']} en entrée, {s['tokens_sortie']} en sortie")
    typer.echo(f"Reranks Pinecone ce mois : {s['reranks_pinecone_ce_mois']}")
    typer.echo(f"Par canal : {s['par_canal']}")
    typer.echo(f"Par intention : {s['par_intention']}")
    for day in s["par_jour"]:
        typer.echo(f"  {day['jour']} : {day['n']} questions, {day['u']} utilisateurs")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Lance l'API."""
    import uvicorn

    get_settings()
    uvicorn.run("rag_fisc_v2.main:app", host=host, port=port, reload=reload)
