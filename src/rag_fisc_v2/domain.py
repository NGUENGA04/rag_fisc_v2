"""Types métier partagés par l'ingestion, la recherche et l'API."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DocType(StrEnum):
    CGI = "cgi"
    LFI = "lfi"
    CIRCULAIRE = "circulaire"
    ARRETE = "arrete"
    FICHE = "fiche"
    BAREME = "bareme"


# Libellés lisibles, utilisés dans les citations.
DOC_TYPE_LABELS: dict[DocType, str] = {
    DocType.CGI: "Code Général des Impôts",
    DocType.LFI: "Loi de finances",
    DocType.CIRCULAIRE: "Circulaire d'application",
    DocType.ARRETE: "Arrêté",
    DocType.FICHE: "Fiche pratique DGI",
    DocType.BAREME: "Barème",
}

# Ordre d'autorité utilisé pour départager des passages de pertinence proche.
DOC_TYPE_AUTHORITY: dict[DocType, int] = {
    DocType.CGI: 0,
    DocType.LFI: 1,
    DocType.ARRETE: 2,
    DocType.CIRCULAIRE: 3,
    DocType.BAREME: 4,
    DocType.FICHE: 5,
}

# Valeur stockée dans Pinecone pour un document sans exercice précis.
EXERCICE_NON_DATE = 0


class Profil(StrEnum):
    EXPERT = "expert"
    GRAND_PUBLIC = "grand_public"


class Canal(StrEnum):
    API = "api"
    WHATSAPP = "whatsapp"
    CLI = "cli"


@dataclass
class LlmUsage:
    """Cumul des jetons consommés pour une question, pour le suivi des coûts."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)


@dataclass
class Chunk:
    """Morceau de document prêt à être vectorisé."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
