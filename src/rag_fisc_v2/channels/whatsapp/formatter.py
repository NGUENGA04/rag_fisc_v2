"""Mise en forme des réponses pour WhatsApp.

WhatsApp n'affiche ni tableaux ni Markdown standard. Il reconnaît *gras*,
_italique_ et ~barré~, et limite un message à 4 096 caractères.
"""

import re

from rag_fisc_v2.domain import DocType, Profil
from rag_fisc_v2.schemas.api import AskResponse, Source

_CITATION = re.compile(r"\[(S\d+|B)\]")


def markdown_to_whatsapp(text: str) -> str:
    lines_out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Titres Markdown -> gras
        m = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if m:
            lines_out.append(f"*{m.group(1).strip('* ')}*")
            continue
        # Lignes de séparation de tableau -> supprimées
        if re.match(r"^\|?[\s:|-]+\|[\s:|-]*$", stripped):
            continue
        # Lignes de tableau -> liste "a : b : c"
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            lines_out.append("- " + " : ".join(c for c in cells if c))
            continue
        # Puces Markdown "* x" -> "- x" (l'astérisque seule passerait pour du gras)
        line = re.sub(r"^(\s*)[*+]\s+", r"\1- ", line)
        lines_out.append(line)
    out = "\n".join(lines_out)
    out = re.sub(r"\*\*(.+?)\*\*", r"*\1*", out)  # gras
    out = re.sub(r"__(.+?)__", r"_\1_", out)
    out = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", out)  # liens
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def short_source_label(s: Source) -> str:
    year = f" {s.exercice}" if s.exercice else ""
    names = {
        DocType.CGI: f"CGI{year}",
        DocType.LFI: f"Loi de finances{year}",
        DocType.CIRCULAIRE: f"Circulaire LF{year}",
        DocType.ARRETE: f"Arrêté{year}",
        DocType.FICHE: "Fiche pratique DGI",
        DocType.BAREME: "Barème DGI",
    }
    parts = [names.get(s.type, s.titre_document)]
    if s.article:
        parts.append(f"art. {s.article}" if s.type != DocType.LFI else f"article {s.article} de la loi")
    if s.type == DocType.LFI and s.articles_cites:
        parts.append(f"sur l'art. {s.articles_cites[0]} du CGI")
    if s.page_debut:
        parts.append(f"p. {s.page_debut}")
    return ", ".join(parts)


def renumber_citations(text: str, sources: list[Source], has_calcul: bool) -> tuple[str, list[str]]:
    """Remplace [S7], [B] par [1], [2] dans l'ordre d'apparition, et construit la liste des sources."""
    by_ref = {s.ref: s for s in sources}
    numbering: dict[str, int] = {}
    footer: list[str] = []

    def repl(m: re.Match) -> str:
        ref = m.group(1)
        if ref == "B":
            if not has_calcul:
                return ""
            label = "Barème mensuel officiel (DGI)"
        elif ref in by_ref:
            label = short_source_label(by_ref[ref])
        else:
            return ""  # référence inventée par le modèle : supprimée
        if ref not in numbering:
            numbering[ref] = len(numbering) + 1
            footer.append(f"[{numbering[ref]}] {label}")
        return f"[{numbering[ref]}]"

    text = _CITATION.sub(repl, text)
    text = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", text)  # doublons consécutifs
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    return text, footer


def split_message(text: str, max_chars: int) -> list[str]:
    """Coupe un texte en messages successifs, de préférence entre deux paragraphes."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        while len(para) > max_chars:  # paragraphe trop long : coupe sur une fin de phrase
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            chunk, para = para[:cut].strip(), para[cut:].strip()
            if current:
                parts.append(current)
                current = ""
            parts.append(chunk)
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars:
            parts.append(current)
            current = para
        else:
            current = candidate
    if current:
        parts.append(current)
    return [p for p in parts if p.strip()]


GRAND_PUBLIC_FOOTER = "_Information indicative tirée des textes officiels. Elle ne remplace pas l'avis d'un professionnel._"


def format_response(response: AskResponse, max_chars: int = 3500) -> list[str]:
    """Réponse du RAG -> liste de messages WhatsApp prêts à envoyer."""
    body, footer = renumber_citations(response.reponse, response.sources, response.calcul is not None)
    body = markdown_to_whatsapp(body)

    blocks = [body]
    if footer:
        blocks.append("📚 *Sources*\n" + "\n".join(footer))
    # Avertissements sur le corpus (CGI manquant, exercice absent, barème non daté).
    corpus_warnings = [w for w in response.avertissements if not w.startswith("Cette réponse est fournie")]
    if corpus_warnings:
        blocks.append("\n".join(f"⚠️ {w}" for w in corpus_warnings))
    if response.profil == Profil.GRAND_PUBLIC and response.sources:
        blocks.append(GRAND_PUBLIC_FOOTER)
    return split_message("\n\n".join(blocks), max_chars)
