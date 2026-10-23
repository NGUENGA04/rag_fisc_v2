"""Nettoyage du Markdown produit par l'OCR, avant découpage."""

import re
from collections import Counter
from dataclasses import dataclass

from rag_fisc_v2.ingestion.ocr import OcrPage
from rag_fisc_v2.ingestion.references import match_cgi_article_header, match_ordinal_article_header


@dataclass
class Line:
    text: str
    page: int


# Lignes de sommaire : "SECTION III : CALCUL ........ 50", "DECLARATIONS...101", "TVA … 107".
_TOC_LINE = re.compile(r"(?:\.{2,}|…+)\s*[\dIVXLC]+\s*\**\s*\|?\s*$|\.{5,}\s*\|?\s*$")
_PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*(?:\d{1,4}|[IVXLC]{1,6})\s*[-–—]?\s*$")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HYPHEN_BREAK = re.compile(r"([a-zà-ÿ])-\n([a-zà-ÿ])")
_MANY_BLANKS = re.compile(r"\n{3,}")


_STRUCTURAL = re.compile(
    r"^[\s#*_]*(?:LIVRE|PARTIE|TITRE|CHAPITRE|SECTION|SOUS-SECTION|PARAGRAPHE|ARTICLE|Article|Art\.)\b"
)


def _is_structural(line: str) -> bool:
    """Lignes qu'il ne faut jamais prendre pour un en-tête de page répété."""
    return bool(
        _STRUCTURAL.match(line)
        or line.lstrip().startswith("|")
        or match_cgi_article_header(line)
        or match_ordinal_article_header(line)
    )


def _normalize_for_count(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def find_running_lines(pages: list[OcrPage], min_ratio: float = 0.3) -> set[str]:
    """Lignes courtes répétées sur une grande part des pages : en-têtes et pieds de page."""
    if len(pages) < 5:
        return set()
    counts: Counter[str] = Counter()
    for p in pages:
        seen = {
            _normalize_for_count(line)
            for line in p.markdown.splitlines()
            if 0 < len(line.strip()) <= 90 and not _is_structural(line)
        }
        counts.update(seen)
    threshold = max(3, int(len(pages) * min_ratio))
    return {line for line, n in counts.items() if n >= threshold and line}


def clean_page_text(text: str) -> str:
    text = _IMAGE.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace(" ", " ").replace("’", "'").replace("", "-")
    return text


def clean_pages(pages: list[OcrPage]) -> list[Line]:
    """Retourne les lignes utiles du document, chacune avec sa page d'origine."""
    running = find_running_lines(pages)
    lines: list[Line] = []
    for p in pages:
        for raw in clean_page_text(p.markdown).splitlines():
            stripped = raw.strip()
            if stripped and (
                _TOC_LINE.search(stripped)
                or _PAGE_NUMBER.match(stripped)
                or _normalize_for_count(stripped) in running
            ):
                continue
            lines.append(Line(raw.rstrip(), p.page))
        lines.append(Line("", p.page))  # une fin de page sépare toujours deux paragraphes

    # Fusion des lignes vides consécutives
    out: list[Line] = []
    for line in lines:
        if not line.text.strip() and out and not out[-1].text.strip():
            continue
        out.append(line)
    return out
