"""Découpage structurel des documents en morceaux indexables.

Trois stratégies, selon le type de document :

- CGI : un morceau par article, avec la hiérarchie Livre > Titre > Chapitre > Section.
  Un article long est découpé par paragraphes, et chaque suite répète l'en-tête.
- Loi de finances : un morceau par article de la loi ("ARTICLE HUITIÈME"). Les
  articles du CGI recopiés dans la loi ouvrent un sous-morceau rattaché à l'article
  du CGI modifié.
- Circulaire, arrêté, fiche : découpage par titres, puis regroupement de paragraphes.

Règles communes : un tableau Markdown n'est jamais coupé, sauf s'il dépasse la
taille limite. Dans ce cas il est coupé par lignes et chaque partie répète l'en-tête.
"""

import re
from dataclasses import dataclass, field

from rag_fisc_v2.domain import Chunk, DocType
from rag_fisc_v2.ingestion.cleaning import Line
from rag_fisc_v2.ingestion.manifest import DocumentSpec
from rag_fisc_v2.ingestion.references import (
    extract_article_refs,
    match_cgi_article_header,
    match_ordinal_article_header,
)

# --- Titres de structure -----------------------------------------------------------

_LEVELS = {
    "PARTIE": 0,
    "LIVRE": 1,
    "TITRE": 2,
    "SOUS-TITRE": 3,
    "CHAPITRE": 4,
    "SECTION": 5,
    "SOUS-SECTION": 6,
    "PARAGRAPHE": 7,
}
_ANNEX_LEVEL = _LEVELS["SOUS-TITRE"]  # une annexe se rattache au titre qui la précède
_STRUCT_HEADING = re.compile(
    r"^[\s#*_]*(?:(?:PREMIERE|DEUXIEME|TROISIEME|QUATRIEME|PREMIÈRE|DEUXIÈME|TROISIÈME|QUATRIÈME)\s+)?"
    r"(?P<kw>PARTIE|LIVRE|SOUS-TITRE|TITRE|CHAPITRE|SOUS-SECTION|SECTION|PARAGRAPHE)\b(?P<rest>.*)$",
    re.IGNORECASE,
)
# "LIVRE DEUXIEME", "TITRE TROISIEME", "CHAPITRE UNIQUE"
_ORDINAL_WORDS = (
    r"(?:PREMI[EÈ]RE?|DEUXI[EÈ]ME|TROISI[EÈ]ME|QUATRI[EÈ]ME|CINQUI[EÈ]ME|SIXI[EÈ]ME|SEPTI[EÈ]ME"
    r"|HUITI[EÈ]ME|NEUVI[EÈ]ME|DIXI[EÈ]ME|ONZI[EÈ]ME|DOUZI[EÈ]ME|UNIQUE)"
)
_ANNEX_HEADING = re.compile(r"^[\s#*_]*(ANNEXES?)\b(?P<rest>.*)$")

# Textes d'application reproduits à la fin du CGI : décisions, circulaires, instructions.
# Leurs articles ont leur propre numérotation et ne sont pas des articles du CGI.
_APPLICATION_TEXT = re.compile(
    r"^[\s#*_>]*(?:D[ÉE]CISION|LETTRE[- ]CIRCULAIRE|CIRCULAIRE|INSTRUCTION(?:\s+INTERMINIST[ÉE]RIELLE)?"
    r"|NOTE\s+DE\s+SERVICE|ARR[ÊE]T[ÉE]|D[ÉE]CRET|COMMUNIQU[ÉE]|LOI|ORDONNANCE)\s+N\s?°",
    re.IGNORECASE,
)
APPLICATION_TEXTS_LABEL = "TEXTES D'APPLICATION"
_ORDINAL_PARTIE = re.compile(r"^[\s#*_]*(PREMI[EÈ]RE|DEUXI[EÈ]ME|TROISI[EÈ]ME|QUATRI[EÈ]ME)\s+PARTIE", re.I)

# Plan des circulaires : "I- ...", "A. ...", "## Titre"
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_OUTLINE_HEADING = re.compile(r"^[\s*_]*(?P<mark>[IVX]{1,5}|[A-H])\s*[-.)–]\s+(?P<title>\S.{2,140})$")


def _clean_heading(text: str) -> str:
    return re.sub(r"[#*_]+", "", text).strip(" :-–.")


def match_struct_heading(line: str) -> tuple[int, str] | None:
    """(niveau, libellé) pour une ligne "TITRE I : IMPÔT SUR LES SOCIÉTÉS"."""
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return None
    if _ORDINAL_PARTIE.match(stripped):
        return 0, _clean_heading(stripped)
    m = _STRUCT_HEADING.match(stripped)
    if not m:
        return None
    rest = m.group("rest").strip()
    # "Section" suivi d'un texte en minuscules est une phrase, pas un titre.
    if rest and not re.match(r"^[\s*_]*([IVXLC\d]+|" + _ORDINAL_WORDS + r"|[Pp]remi[eè]re?|[Uu]nique|[A-Z])\b", rest):
        return None
    if stripped[0].islower():
        return None
    kw = m.group("kw").upper()
    return _LEVELS[kw], _clean_heading(stripped)


def match_outline_heading(line: str) -> tuple[int, str] | None:
    """Titres de circulaire : Markdown (#) ou numérotation I-, A-."""
    stripped = line.strip()
    m = _MD_HEADING.match(stripped)
    if m:
        return len(m.group(1)), _clean_heading(m.group(2))
    m = _OUTLINE_HEADING.match(stripped)
    if m and not stripped.endswith((";", ",")):
        mark = m.group("mark")
        level = 2 if re.fullmatch(r"[IVX]+", mark) else 3
        return level, _clean_heading(stripped)
    return None


# --- Blocs : paragraphes et tableaux ------------------------------------------------


@dataclass
class Block:
    text: str
    page_start: int
    page_end: int
    is_table: bool = False


def lines_to_blocks(lines: list[Line]) -> list[Block]:
    """Regroupe les lignes en paragraphes et en tableaux."""
    blocks: list[Block] = []
    buf: list[Line] = []
    buf_table = False

    def flush():
        nonlocal buf
        text = "\n".join(line.text for line in buf).strip()
        if text:
            blocks.append(Block(text, buf[0].page, buf[-1].page, buf_table))
        buf = []

    for line in lines:
        is_table_line = line.text.lstrip().startswith("|")
        if not line.text.strip():
            if not buf_table:
                flush()
            continue
        if buf and is_table_line != buf_table:
            flush()
        buf_table = is_table_line
        buf.append(line)
    flush()
    return blocks


def split_table(block: Block, max_chars: int) -> list[Block]:
    """Coupe un tableau trop long par lignes, en répétant les lignes d'en-tête."""
    rows = block.text.splitlines()
    head = rows[:2] if len(rows) > 2 and re.match(r"^\s*\|[\s:|-]+\|\s*$", rows[1]) else rows[:1]
    body = rows[len(head):]
    parts, current = [], list(head)
    for row in body:
        if len("\n".join(current + [row])) > max_chars and len(current) > len(head):
            parts.append(current)
            current = list(head)
        current.append(row)
    parts.append(current)
    return [Block("\n".join(p), block.page_start, block.page_end, True) for p in parts]


def split_long_paragraph(block: Block, max_chars: int) -> list[Block]:
    """Coupe un paragraphe trop long sur les fins de phrases."""
    sentences = re.split(r"(?<=[.;:])\s+", block.text)
    parts, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return [Block(p, block.page_start, block.page_end) for p in parts]


def pack_blocks(blocks: list[Block], max_chars: int, header: str | None = None) -> list[list[Block]]:
    """Regroupe des blocs en morceaux d'au plus max_chars, sans couper un tableau."""
    budget = max_chars - (len(header) + 2 if header else 0)
    expanded: list[Block] = []
    for b in blocks:
        if len(b.text) <= budget:
            expanded.append(b)
        elif b.is_table:
            expanded.extend(split_table(b, budget))
        else:
            expanded.extend(split_long_paragraph(b, budget))

    groups: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for b in expanded:
        if current and size + len(b.text) + 2 > budget:
            groups.append(current)
            current, size = [], 0
        current.append(b)
        size += len(b.text) + 2
    if current:
        groups.append(current)
    return groups


# --- Unités logiques -----------------------------------------------------------------


@dataclass
class Unit:
    """Portion de document avant le découpage par taille : un article, une section."""

    hierarchy: list[str]
    lines: list[Line] = field(default_factory=list)
    article: str | None = None  # numéro dans le document lui-même
    linked_article: str | None = None  # article du CGI recopié dans une loi de finances
    header: str | None = None  # libellé répété en tête des suites


def _hierarchy_label(stack: dict[int, str]) -> list[str]:
    return [stack[k] for k in sorted(stack)]


def _set_level(stack: dict[int, str], level: int, label: str) -> None:
    for k in [k for k in stack if k >= level]:
        del stack[k]
    stack[level] = label


def _heading_with_continuation(lines: list[Line], i: int, label: str) -> tuple[str, int]:
    """Ajoute au titre sa ligne de libellé quand elle est sur la ligne suivante."""
    j = i + 1
    while j < len(lines) and not lines[j].text.strip():
        j += 1
    if j < len(lines):
        nxt = lines[j].text.strip()
        is_caps_title = nxt.isupper() and len(nxt) < 150
        if (
            is_caps_title
            and not match_struct_heading(nxt)
            and not match_cgi_article_header(nxt)
            and not match_ordinal_article_header(nxt)
            and not nxt.startswith("|")
            and len(label) < 40
        ):
            return f"{label} : {_clean_heading(nxt)}", j
    return label, i


def units_cgi(lines: list[Line]) -> list[Unit]:
    units: list[Unit] = []
    stack: dict[int, str] = {}
    current = Unit(hierarchy=[])
    # Annexes du Code ("ANNEXE 1 : PARTIE LEGISLATIVE", "ANNEXE 2 : PARTIE REGLEMENTAIRE") : une fois
    # entré, on y reste jusqu'à la fin du document. Elles reproduisent des lois, ordonnances, décrets
    # et circulaires entiers, dont les articles ne sont pas des articles du CGI.
    zone: str | None = None
    application_text: str | None = None  # titre du texte reproduit en cours, s'il y en a un
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.text.strip()

        opened = _annex_zone_at(lines, i)
        if opened:
            zone, i = opened
            application_text = None
            stack = {0: ANNEXES_LABEL, 1: zone}
            units.append(current)
            current = Unit(hierarchy=_hierarchy_label(stack))
            i += 1
            continue

        if _is_application_text_title(stripped):
            application_text = _clean_heading(stripped)[:160]
            stack = {0: ANNEXES_LABEL, 1: zone, 2: application_text} if zone else {
                0: APPLICATION_TEXTS_LABEL, 1: application_text
            }
            units.append(current)
            current = Unit(hierarchy=_hierarchy_label(stack), header=application_text)
            current.lines.append(line)
            i += 1
            continue

        heading = match_struct_heading(line.text)
        annex = _ANNEX_HEADING.match(stripped) if not heading else None
        if heading or annex:
            if heading:
                level, label = heading
                if level <= _LEVELS["LIVRE"] and application_text and not zone:
                    application_text = None  # retour au corps du Code : le plan repart de zéro
                    stack = {}
            else:
                level, label = _ANNEX_LEVEL, _clean_heading(stripped)
            label, i = _heading_with_continuation(lines, i, label)
            if zone:
                level += 3  # sous "ANNEXES DU CGI > zone > texte reproduit"
            elif application_text:
                level += 2  # le plan d'un texte d'application se range sous son titre
            _set_level(stack, level, label)
            units.append(current)
            current = Unit(hierarchy=_hierarchy_label(stack), header=application_text)
            i += 1
            continue

        article = match_cgi_article_header(line.text)
        if article:
            units.append(current)
            source = application_text or zone
            if source:
                # Article d'un texte reproduit en annexe : pas un article du CGI.
                current = Unit(hierarchy=_hierarchy_label(stack), header=f"{source[:100]}, article {article}")
            else:
                current = Unit(hierarchy=_hierarchy_label(stack), article=article, header=f"Article {article}")
        current.lines.append(line)
        i += 1
    units.append(current)
    return units


ANNEXES_LABEL = "ANNEXES DU CGI"
_ANNEX_ZONE = re.compile(
    r"^[\s#*_]*(?:ANNEXE\s+\d+[\s*_:–-]*)?PARTIE\s+(?P<zone>L[ÉE]GISLATIVE|R[ÉE]GLEMENTAIRE)\b", re.IGNORECASE
)
_ANNEX_NUMBER_ONLY = re.compile(r"^[\s#*_]*ANNEXE\s+\d+[\s*_:]*$", re.IGNORECASE)


def _annex_zone_at(lines: list[Line], i: int) -> tuple[str, int] | None:
    """Ouverture d'une partie annexe : "ANNEXE 1 PARTIE LEGISLATIVE", éventuellement sur deux lignes.

    Retourne le nom de la zone et l'index de la dernière ligne consommée.
    """
    stripped = lines[i].text.strip()
    m = _ANNEX_ZONE.match(stripped)
    if m:
        return f"PARTIE {m.group('zone').upper()}", i
    if _ANNEX_NUMBER_ONLY.match(stripped):
        j = i + 1
        while j < len(lines) and not lines[j].text.strip():
            j += 1
        if j < len(lines):
            m = _ANNEX_ZONE.match(lines[j].text.strip())
            if m:
                return f"PARTIE {m.group('zone').upper()}", j
    return None


def _is_application_text_title(stripped: str) -> bool:
    """Titre d'un texte d'application : "# **DECISION N° 0000127/MINFI/DGI DU 12 FEVRIER 2016**"."""
    if not _APPLICATION_TEXT.match(stripped) or len(stripped) > 250:
        return False
    # Un titre, pas une phrase qui cite le texte : ligne de titre Markdown, en gras ou en capitales.
    return stripped.startswith(("#", "**")) or _clean_heading(stripped).isupper()


def units_lfi(lines: list[Line]) -> list[Unit]:
    units: list[Unit] = []
    stack: dict[int, str] = {}
    current = Unit(hierarchy=[])
    loi_article: str | None = None
    for line in lines:
        heading = match_struct_heading(line.text)
        if heading:
            _set_level(stack, *heading)
            units.append(current)
            current = Unit(hierarchy=_hierarchy_label(stack), article=loi_article)
            continue
        ordinal = match_ordinal_article_header(line.text)
        if ordinal:
            loi_article = str(ordinal)
            units.append(current)
            current = Unit(
                hierarchy=_hierarchy_label(stack),
                article=loi_article,
                header=f"Loi de finances, article {loi_article}",
            )
        else:
            cgi_article = match_cgi_article_header(line.text)
            if cgi_article and loi_article:
                units.append(current)
                current = Unit(
                    hierarchy=_hierarchy_label(stack),
                    article=loi_article,
                    linked_article=cgi_article,
                    header=f"Loi de finances, article {loi_article} : article {cgi_article} du CGI",
                )
        current.lines.append(line)
    units.append(current)
    return units


def units_sections(lines: list[Line]) -> list[Unit]:
    """Circulaires, arrêtés, fiches : titres du plan et articles numérotés."""
    units: list[Unit] = []
    stack: dict[int, str] = {}
    current = Unit(hierarchy=[])
    for line in lines:
        heading = match_outline_heading(line.text) or match_struct_heading(line.text)
        article = match_cgi_article_header(line.text)
        if heading and not article:
            level, label = heading
            _set_level(stack, level, label)
            units.append(current)
            current = Unit(hierarchy=_hierarchy_label(stack), header=label)
        elif article:
            units.append(current)
            current = Unit(hierarchy=_hierarchy_label(stack), article=article, header=f"Article {article}")
        current.lines.append(line)
    units.append(current)
    return units


# --- Assemblage final ------------------------------------------------------------------


def _meaningful(text: str) -> bool:
    """Écarte les unités réduites à un titre ou à quelques caractères."""
    return len(re.sub(r"[\s#*_|:\-]", "", text)) >= 40


def _merge_small_units(units: list[Unit], min_chars: int, max_chars: int) -> list[Unit]:
    """Fusionne les petites sections voisines sans article (plan des circulaires)."""
    merged: list[Unit] = []
    for u in units:
        size = sum(len(line.text) for line in u.lines)
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and u.article is None
            and prev.article is None
            and prev.linked_article is None
            and prev.hierarchy[:1] == u.hierarchy[:1]
            and (size < min_chars or sum(len(line.text) for line in prev.lines) < min_chars)
            and sum(len(line.text) for line in prev.lines) + size < max_chars
        ):
            prev.lines.extend(u.lines)
            continue
        merged.append(u)
    return merged


def _update_subtitles(subtitles: dict[int, str], block: Block) -> None:
    """Met à jour la pile des sous-titres internes d'un article avec un bloc titre Markdown."""
    if block.is_table or "\n" in block.text:
        return
    m = _MD_HEADING.match(block.text.strip())
    if not m or match_cgi_article_header(block.text):
        return
    title = _clean_heading(m.group(2))
    level = _subtitle_level(title, len(m.group(1)))
    for k in [k for k in subtitles if k >= level]:
        del subtitles[k]
    subtitles[level] = title


def _subtitle_level(title: str, markdown_level: int) -> int:
    """Niveau d'après la numérotation, plus fiable que le nombre de # produit par l'OCR.

    "A - Frais généraux" (1) > "1- Rémunérations" (2) > "a) Salaires" (3) > titre non numéroté.
    """
    if re.match(r"^[A-Z]\s*[-–.)]", title):
        return 1
    if re.match(r"^\d{1,2}\s*[-–.)°]", title):
        return 2
    if re.match(r"^[a-z]\s*[-–.)]", title):
        return 3
    return 3 + markdown_level


def chunk_document(
    doc: DocumentSpec,
    lines: list[Line],
    max_chars: int = 3500,
    min_chars: int = 300,
) -> list[Chunk]:
    if doc.type == DocType.CGI:
        units = units_cgi(lines)
    elif doc.type == DocType.LFI:
        units = units_lfi(lines)
    else:
        units = _merge_small_units(units_sections(lines), min_chars, max_chars)

    chunks: list[Chunk] = []
    for unit in units:
        blocks = lines_to_blocks(unit.lines)
        full_text = "\n\n".join(b.text for b in blocks)
        if not blocks or not _meaningful(full_text):
            continue
        # Réserve de place pour l'en-tête des suites, sous-titres compris.
        groups = pack_blocks(blocks, max_chars - 120, unit.header)
        subtitles: dict[int, str] = {}  # sous-titres internes : "A - Frais généraux" > "1- Rémunérations"
        for part, group in enumerate(groups):
            body = "\n\n".join(b.text for b in group)
            if part > 0 and unit.header:
                context = " > ".join(subtitles[k] for k in sorted(subtitles))
                suffix = f" — {context[-150:]}" if context else ""
                body = f"{unit.header} (suite){suffix}\n\n{body}"
            for b in group:
                _update_subtitles(subtitles, b)
            refs = extract_article_refs(body)
            if unit.article and doc.type != DocType.LFI and unit.article in refs:
                refs.remove(unit.article)  # l'en-tête de l'article n'est pas un renvoi
            if unit.linked_article and unit.linked_article not in refs:
                refs.insert(0, unit.linked_article)
            metadata = {
                "doc_id": doc.doc_id,
                "type": doc.type.value,
                "exercice": doc.exercice_meta,
                "titre_document": doc.titre,
                "juridiction": "cameroun",
                "hierarchie": " > ".join(unit.hierarchy),
                "page_debut": group[0].page_start,
                "page_fin": group[-1].page_end,
                "partie": part,
            }
            if unit.article:
                metadata["article"] = unit.article
            if refs:
                metadata["articles_cites"] = refs[:30]
            if doc.reference:
                metadata["reference"] = doc.reference
            chunks.append(
                Chunk(chunk_id=f"{doc.doc_id}#{len(chunks):05d}", text=body, metadata=metadata)
            )
    return chunks
