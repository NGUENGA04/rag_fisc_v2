"""Enchaînement des étapes d'ingestion pour un ou plusieurs documents."""

import json
import logging
from dataclasses import asdict, dataclass

from rag_fisc_v2.config import get_settings
from rag_fisc_v2.domain import Chunk, DocType
from rag_fisc_v2.ingestion.baremes import extract_bareme, save_bareme
from rag_fisc_v2.ingestion.chunking import chunk_document
from rag_fisc_v2.ingestion.cleaning import clean_pages
from rag_fisc_v2.ingestion.manifest import DocumentSpec, Manifest
from rag_fisc_v2.ingestion.ocr import OcrDocument, OcrPage, export_markdown, run_ocr

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    doc_id: str
    pages: int = 0
    chunks: int = 0
    indexed: int = 0
    note: str = ""


def text_layer_pages(doc: DocumentSpec) -> OcrDocument:
    """Texte natif du PDF, pour un document marqué ocr: false."""
    from pypdf import PdfReader

    reader = PdfReader(doc.path())
    pages = [OcrPage(page=i + 1, markdown=p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return OcrDocument(doc_id=doc.doc_id, source_sha256=doc.sha256(), model="pypdf", pages=pages)


def chunks_path(doc_id: str):
    return get_settings().chunks_dir / f"{doc_id}.jsonl"


def save_chunks(doc_id: str, chunks: list[Chunk]) -> None:
    path = chunks_path(doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(doc_id: str) -> list[Chunk]:
    path = chunks_path(doc_id)
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def build_chunks(doc: DocumentSpec, force_ocr: bool = False) -> tuple[OcrDocument, list[Chunk]]:
    settings = get_settings()
    ocr = run_ocr(doc, force=force_ocr) if doc.ocr else text_layer_pages(doc)
    export_markdown(ocr, settings.ocr_dir / f"{doc.doc_id}.md")
    lines = clean_pages(ocr.pages)
    chunks = chunk_document(doc, lines, settings.chunk_max_chars, settings.chunk_min_chars)
    save_chunks(doc.doc_id, chunks)
    return ocr, chunks


def ingest_document(
    doc: DocumentSpec,
    *,
    index: bool = True,
    force_ocr: bool = False,
    force_index: bool = False,
) -> IngestReport:
    report = IngestReport(doc_id=doc.doc_id)

    if doc.type == DocType.BAREME:
        data = extract_bareme(doc)
        save_bareme(data)
        report.note = f"barème extrait : {len(data['lignes'])} lignes"
        return report

    ocr, chunks = build_chunks(doc, force_ocr=force_ocr)
    report.pages = len(ocr.pages)
    report.chunks = len(chunks)

    if index and doc.indexer:
        from rag_fisc_v2.ingestion.indexer import index_chunks

        report.indexed = index_chunks(doc.doc_id, chunks, force=force_index)
    elif not doc.indexer:
        report.note = "exclu de l'index (indexer: false)"
    return report


def ingest(
    manifest: Manifest,
    doc_ids: list[str] | None = None,
    **kwargs,
) -> list[IngestReport]:
    missing = [d.fichier for d in manifest.select(doc_ids) if not d.path().exists()]
    if missing:
        raise FileNotFoundError(f"PDF introuvables dans data/raw : {missing}")
    reports = []
    for doc in manifest.select(doc_ids):
        log.info("=== %s", doc.doc_id)
        reports.append(ingest_document(doc, **kwargs))
    return reports
