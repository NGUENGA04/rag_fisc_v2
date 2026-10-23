"""OCR des PDF avec Mistral OCR, avec cache sur disque.

Chaque PDF est découpé en lots de quelques pages. Chaque lot est envoyé à
l'API puis mis en cache séparément. Une interruption reprend donc là où elle
s'est arrêtée, et un document déjà traité n'est jamais repayé.

Cache :
    data/ocr/<doc_id>/part_000.json   résultat brut d'un lot
    data/ocr/<doc_id>.json            document assemblé (pages en Markdown)
"""

import io
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from rag_fisc_v2.clients import get_mistral
from rag_fisc_v2.config import get_settings
from rag_fisc_v2.ingestion.manifest import DocumentSpec

log = logging.getLogger(__name__)


@dataclass
class OcrPage:
    page: int  # numéro de page dans le PDF d'origine, à partir de 1
    markdown: str
    header: str | None = None
    footer: str | None = None


@dataclass
class OcrDocument:
    doc_id: str
    source_sha256: str
    model: str
    pages: list[OcrPage]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OcrDocument":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["pages"] = [OcrPage(**p) for p in data["pages"]]
        return cls(**data)


def ocr_cache_path(doc_id: str) -> Path:
    return get_settings().ocr_dir / f"{doc_id}.json"


def load_cached_ocr(doc_id: str) -> OcrDocument | None:
    path = ocr_cache_path(doc_id)
    return OcrDocument.load(path) if path.exists() else None


_TABLE_LINK = re.compile(r"\[(tbl-\d+\.(?:md|html))\]\(\1\)")


def _inline_tables(markdown: str, tables: list) -> str:
    """Remplace les liens vers les tableaux extraits par leur contenu."""
    if not tables:
        return markdown
    by_id = {t.id: t.content for t in tables}
    return _TABLE_LINK.sub(lambda m: by_id.get(m.group(1), m.group(0)), markdown)


def _str_or_none(value) -> str | None:
    """Le SDK renvoie une sentinelle UNSET pour les champs absents."""
    return value if isinstance(value, str) and value.strip() else None


def _split_pdf(path: Path, pages_per_part: int) -> list[tuple[int, bytes]]:
    """Retourne (index de la première page, octets du sous-PDF) pour chaque lot."""
    reader = PdfReader(path)
    parts: list[tuple[int, bytes]] = []
    for start in range(0, len(reader.pages), pages_per_part):
        writer = PdfWriter()
        for i in range(start, min(start + pages_per_part, len(reader.pages))):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        parts.append((start, buf.getvalue()))
    return parts


def _ocr_part(content: bytes, file_name: str, model: str) -> list[dict]:
    """Envoie un lot à Mistral OCR via l'API Files et retourne ses pages."""
    client = get_mistral()
    uploaded = client.files.upload(
        file={"file_name": file_name, "content": content},
        purpose="ocr",
    )
    try:
        signed = client.files.get_signed_url(file_id=uploaded.id)
        response = client.ocr.process(
            model=model,
            document={"type": "document_url", "document_url": signed.url},
            table_format="markdown",
            extract_header=True,
            extract_footer=True,
            include_image_base64=False,
            include_blocks=False,
        )
    finally:
        try:
            client.files.delete(file_id=uploaded.id)
        except Exception:  # le nettoyage ne doit pas masquer l'erreur principale
            log.warning("Fichier %s non supprimé chez Mistral", uploaded.id)

    pages = []
    for p in response.pages:
        pages.append(
            {
                "index": p.index,
                "markdown": _inline_tables(p.markdown or "", getattr(p, "tables", None) or []),
                "header": _str_or_none(getattr(p, "header", None)),
                "footer": _str_or_none(getattr(p, "footer", None)),
            }
        )
    return pages


def run_ocr(doc: DocumentSpec, force: bool = False) -> OcrDocument:
    """OCR complet d'un document, en réutilisant le cache quand c'est possible."""
    settings = get_settings()
    sha = doc.sha256()
    final_path = ocr_cache_path(doc.doc_id)

    if not force and final_path.exists():
        cached = OcrDocument.load(final_path)
        if cached.source_sha256 == sha:
            log.info("%s : OCR en cache", doc.doc_id)
            return cached
        log.info("%s : le PDF a changé, nouvel OCR", doc.doc_id)

    parts_dir = settings.ocr_dir / doc.doc_id
    parts_dir.mkdir(parents=True, exist_ok=True)
    pages: list[OcrPage] = []

    for n, (start, content) in enumerate(_split_pdf(doc.path(), settings.ocr_pages_per_part)):
        part_path = parts_dir / f"part_{n:03d}.json"
        cached_part = None
        if not force and part_path.exists():
            cached_part = json.loads(part_path.read_text(encoding="utf-8"))
            if cached_part.get("source_sha256") != sha or cached_part.get("model") != settings.ocr_model:
                cached_part = None

        if cached_part is None:
            log.info("%s : OCR du lot %d (pages %d et suivantes)", doc.doc_id, n, start + 1)
            part_pages = _ocr_part(content, f"{doc.doc_id}_part{n:03d}.pdf", settings.ocr_model)
            cached_part = {"source_sha256": sha, "model": settings.ocr_model, "start": start, "pages": part_pages}
            part_path.write_text(json.dumps(cached_part, ensure_ascii=False), encoding="utf-8")

        for p in cached_part["pages"]:
            pages.append(
                OcrPage(
                    page=start + p["index"] + 1,
                    markdown=p["markdown"],
                    header=p.get("header"),
                    footer=p.get("footer"),
                )
            )

    result = OcrDocument(doc_id=doc.doc_id, source_sha256=sha, model=settings.ocr_model, pages=pages)
    result.save(final_path)
    return result


def export_markdown(ocr: OcrDocument, path: Path) -> None:
    """Écrit le texte OCR dans un fichier Markdown lisible, pour relecture humaine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in ocr.pages:
            f.write(f"\n\n<!-- page {p.page} -->\n\n{p.markdown}")
