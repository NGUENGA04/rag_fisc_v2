"""Détection et normalisation des numéros d'articles.

Formes rencontrées dans le corpus camerounais :
    CGI         "Article 7", "Article 7 bis", "ARTICLE 92 BIS", "Art. L106", "Article C 12"
    LFI         "ARTICLE HUITIÈME", "Article premier", "Article vingt-et-unième"
    Renvois     "articles 21, 87, 92, 92 bis, 143 et 149 du Code Général des Impôts"

Forme normalisée : préfixe en majuscule collé au numéro, suffixe latin en
minuscules séparé par une espace. Exemples : "7", "7 bis", "L106", "C12".
"""

import re
import unicodedata

LATIN_SUFFIXES = (
    "bis",
    "ter",
    "quater",
    "quinquies",
    "sexies",
    "septies",
    "octies",
    "nonies",
    "novies",
    "decies",
)
_SUFFIX = r"(?:" + "|".join(LATIN_SUFFIXES) + r")"

# Numéro d'article du CGI : préfixe facultatif (L = procédures fiscales,
# C = fiscalité locale, M = autres livres), nombre ou "premier"/"1er", suffixe facultatif.
_NUMBER = (
    r"(?P<prefix>[LCM])?\s?\.?\s?"
    r"(?P<num>\d{1,4}|premier|1\s?er)"
    r"(?:\s*[-\s]\s*(?P<suffix>" + _SUFFIX + r"))?"
)
_NUMBER_RE = re.compile(_NUMBER + r"\b", re.IGNORECASE)


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize_article(prefix: str | None, num: str, suffix: str | None = None) -> str:
    num = num.lower().replace(" ", "")
    if num in {"premier", "1er"}:
        num = "1"
    num = str(int(num))
    out = f"{(prefix or '').upper()}{num}"
    if suffix:
        out += f" {suffix.lower()}"
    return out


def parse_article_id(text: str) -> str | None:
    """Normalise une chaîne comme "92 BIS", "L. 106" ou "Article 17" ; None si ce n'est pas un numéro."""
    text = re.sub(r"^\s*(?:articles?|art\.?)\s+", "", text, flags=re.IGNORECASE)
    m = _NUMBER_RE.fullmatch(text.strip().rstrip(".-–: "))
    if not m:
        m = _NUMBER_RE.match(text.strip())
        if not m:
            return None
    return normalize_article(m.group("prefix"), m.group("num"), m.group("suffix"))


# --- Ordinaux en toutes lettres (lois de finances) ------------------------------

_UNITS = {
    "premier": 1, "premiere": 1, "un": 1, "unieme": 1, "deux": 2, "deuxieme": 2, "second": 2, "seconde": 2,
    "trois": 3, "troisieme": 3, "quatre": 4, "quatrieme": 4, "cinq": 5, "cinquieme": 5,
    "six": 6, "sixieme": 6, "sept": 7, "septieme": 7, "huit": 8, "huitieme": 8,
    "neuf": 9, "neuvieme": 9, "dix": 10, "dixieme": 10, "onze": 11, "onzieme": 11,
    "douze": 12, "douzieme": 12, "treize": 13, "treizieme": 13, "quatorze": 14,
    "quatorzieme": 14, "quinze": 15, "quinzieme": 15, "seize": 16, "seizieme": 16,
}
_TENS = {
    "vingt": 20, "vingtieme": 20, "trente": 30, "trentieme": 30, "quarante": 40,
    "quarantieme": 40, "cinquante": 50, "cinquantieme": 50, "soixante": 60,
    "soixantieme": 60, "quatre-vingt": 80, "quatre-vingts": 80, "quatre-vingtieme": 80,
    "cent": 100, "centieme": 100,
}
_ORDINAL_WORD = r"[a-zA-ZÀ-ÿ]+(?:[-\s](?:et[-\s])?[a-zA-ZÀ-ÿ]+){0,4}"


def parse_french_ordinal(words: str) -> int | None:
    """Convertit "huitième", "vingt-et-unième", "soixante-douzième" en entier."""
    w = strip_accents(words.lower()).strip()
    w = w.replace("quatre-vingt", "quatre_vingt").replace("quatre vingt", "quatre_vingt")
    tokens = [t for t in re.split(r"[-\s]+", w) if t and t != "et"]
    if not tokens:
        return None
    total = 0
    for t in tokens:
        t = t.replace("quatre_vingt", "quatre-vingt")
        if t in _TENS:
            total += _TENS[t]
        elif t in _UNITS:
            total += _UNITS[t]
        elif t.rstrip("s") in _TENS:
            total += _TENS[t.rstrip("s")]
        else:
            return None
    # Toutes les formes doivent finir par un ordinal ("ieme", "premier") pour éviter
    # de prendre un mot ordinaire pour un numéro.
    last = tokens[-1]
    if not (last.endswith("ieme") or last in {"premier", "premiere", "second", "seconde"}):
        return None
    return total or None


# --- En-têtes d'articles -----------------------------------------------------------

# Début de ligne : décorations Markdown possibles (#, *, _, >, espaces).
_LINE_START = r"^[\s#>*_|]*"
_ARTICLE_WORD = r"(?:ARTICLE|Article|Art\.)"
# Après le numéro : un séparateur typique d'en-tête, ou la fin de ligne.
# "Article 5,-" (virgule au lieu du point) est une erreur d'OCR fréquente.
_HEADER_END = r"[\s*_]*(?:[.\-–—:(]|,\s*[-–]|$)"

_CGI_HEADER_RE = re.compile(
    _LINE_START + _ARTICLE_WORD + r"\s+" + _NUMBER + _HEADER_END, re.MULTILINE | re.IGNORECASE
)
_ORDINAL_HEADER_RE = re.compile(
    _LINE_START + r"(?:ARTICLE|Article)\s+(?P<ordinal>" + _ORDINAL_WORD + r")" + _HEADER_END,
    re.MULTILINE,
)


def match_cgi_article_header(line: str) -> str | None:
    """Numéro normalisé si la ligne commence par un en-tête d'article numéroté."""
    m = _CGI_HEADER_RE.match(line)
    if not m:
        return None
    return normalize_article(m.group("prefix"), m.group("num"), m.group("suffix"))


def match_ordinal_article_header(line: str) -> int | None:
    """Numéro si la ligne commence par "ARTICLE HUITIÈME" ou "Article premier"."""
    m = _ORDINAL_HEADER_RE.match(line)
    if not m:
        return None
    words = m.group("ordinal")
    # Retirer les mots qui suivent l'ordinal sur la même ligne ("ARTICLE TROISIEME Relèvement").
    candidates = re.split(r"\s+", words)
    for n in range(len(candidates), 0, -1):
        value = parse_french_ordinal(" ".join(candidates[:n]))
        if value is not None:
            return value
    return None


# --- Renvois à des articles dans le texte ------------------------------------------

_REF_HEAD = re.compile(r"\b(?:articles?|art\.)\s+", re.IGNORECASE)
# Un numéro de la liste, suivi éventuellement de son alinéa : "22(3)", "8 (1) a)", "7 alinéa 2".
_REF_ITEM = re.compile(
    r"\s*(?:,|et|à|au|ou|&)?\s*"
    + _NUMBER
    + r"\b(?:\s*\(\s*\d+\s*\)(?:\s*-?\s*[a-z]\))?|\s+alin[ée]a\s+\d+)?",
    re.IGNORECASE,
)
_NOT_CGI_CONTEXT = re.compile(
    r"^[^.;\n]{0,40}\b(?:"
    r"de la (?:pr[ée]sente )?loi|du (?:pr[ée]sent )?d[ée]cret|de l'arr[êe]t[ée]|du pr[ée]sent arr[êe]t[ée]"
    r"|de la pr[ée]sente (?:circulaire|note|instruction|d[ée]cision)|du trait[ée]|de la directive"
    r"|du r[èe]glement|de l'acte uniforme|du code des douanes|du code du travail|du code p[ée]nal"
    r")",
    re.IGNORECASE,
)


def _is_header_position(text: str, pos: int) -> bool:
    """Vrai si le mot "Article" en position pos ouvre une ligne d'en-tête d'article."""
    line_start = text.rfind("\n", 0, pos) + 1
    if text[line_start:pos].strip(" \t#>*_|"):
        return False
    line_end = text.find("\n", pos)
    line = text[line_start : line_end if line_end != -1 else len(text)]
    return match_cgi_article_header(line) is not None


def extract_article_refs(text: str) -> list[str]:
    """Liste ordonnée et sans doublon des articles cités dans un texte.

    Les renvois explicitement rattachés à un autre texte ("article 3 de la
    présente loi", "article 12 du décret") sont ignorés.
    """
    refs: list[str] = []
    for head in _REF_HEAD.finditer(text):
        if _is_header_position(text, head.start()):
            continue  # en-tête d'article ("Article 9.- ..."), pas un renvoi
        pos = head.end()
        found: list[str] = []
        while True:
            m = _REF_ITEM.match(text, pos)
            if not m:
                break
            found.append(normalize_article(m.group("prefix"), m.group("num"), m.group("suffix")))
            pos = m.end()
        if found and not _NOT_CGI_CONTEXT.match(text[pos:]):
            for r in found:
                if r not in refs:
                    refs.append(r)
    return refs


_YEAR_RE = re.compile(r"\b(20[0-9]{2})\b")


def extract_years(text: str) -> list[int]:
    return sorted({int(y) for y in _YEAR_RE.findall(text)})
