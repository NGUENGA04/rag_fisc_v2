"""Lecture et normalisation des références [S1], [B] produites par le modèle.

Le modèle ne respecte pas toujours le format demandé. Il écrit parfois
"[S1, art. 17(1)]", "[S1, S3]" ou "[S1; B]". Ces variantes sont ramenées à la
forme canonique "[S1][S3]", la seule que reconnaissent l'API et WhatsApp.
"""

import re

_BRACKET = re.compile(r"\[([^\[\]\n]{1,100})\]")
_STARTS_WITH_REF = re.compile(r"^\s*(?:S\d+|B)\b")
_REF = re.compile(r"\b(S\d+|B)\b")


def _refs_in_group(content: str) -> list[str]:
    if not _STARTS_WITH_REF.match(content):
        return []  # "[voir annexe]" : un crochet ordinaire, pas une référence
    return list(dict.fromkeys(_REF.findall(content)))


def normalize_citations(text: str) -> str:
    def repl(m: re.Match) -> str:
        refs = _refs_in_group(m.group(1))
        return "".join(f"[{r}]" for r in refs) if refs else m.group(0)

    return _BRACKET.sub(repl, text)


def find_citations(text: str) -> list[str]:
    """Références citées, sans doublon, dans l'ordre d'apparition."""
    found: list[str] = []
    for m in _BRACKET.finditer(text):
        for ref in _refs_in_group(m.group(1)):
            if ref not in found:
                found.append(ref)
    return found
