"""Barèmes mensuels (IRPP, TDL) : extraction en tables exactes et recherche.

Ces PDF contiennent des milliers de lignes "salaire -> montant". Les vectoriser
serait inutile : la recherche sémantique ne retrouve pas une ligne chiffrée.
On les extrait donc en JSON, et une question de calcul est résolue par une
recherche exacte dans la table.
"""

import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rag_fisc_v2.config import get_settings
from rag_fisc_v2.ingestion.manifest import DocumentSpec

# Colonnes attendues, dans l'ordre du PDF, pour chaque barème connu.
BAREME_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "bareme-irpp": [
        ("base", "Base mensuelle brute"),
        ("irpp", "IRPP"),
        ("cac", "Centimes additionnels communaux (CAC)"),
        ("ccf", "Crédit foncier, part salariale (CCF)"),
        ("fne_salarial", "FNE, part salariale"),
        ("fne_patronal", "FNE, part patronale"),
        ("crtv", "Redevance audiovisuelle (CRTV)"),
    ],
    "bareme-tdl": [
        ("base", "Salaire de base mensuel"),
        ("tdl", "Taxe de développement local (TDL)"),
    ],
}

# Mots-clés qui orientent une question vers un barème.
BAREME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "bareme-irpp": ("irpp", "cac", "ccf", "fne", "crtv", "impôt sur le revenu", "retenue sur salaire"),
    "bareme-tdl": ("tdl", "taxe de développement local", "développement local"),
}


def _to_int(cell: str | None) -> int:
    digits = re.sub(r"[^\d]", "", cell or "")
    return int(digits) if digits else 0


def extract_bareme(doc: DocumentSpec) -> dict:
    """Lit les tableaux du PDF avec pdfplumber et retourne la table complète."""
    import pdfplumber

    columns = BAREME_COLUMNS[doc.doc_id]
    width = len(columns)
    rows: dict[int, list[int]] = {}
    with pdfplumber.open(doc.path()) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for raw in table:
                    if not raw or len(raw) != width:
                        continue
                    base = _to_int(raw[0])
                    if base == 0:  # ligne d'en-tête
                        continue
                    rows[base] = [base] + [_to_int(c) for c in raw[1:]]

    if not rows:
        raise ValueError(f"{doc.doc_id} : aucun tableau exploitable trouvé")
    ordered = [rows[k] for k in sorted(rows)]
    bases = [r[0] for r in ordered]
    steps = {b - a for a, b in zip(bases, bases[1:], strict=False)}
    return {
        "doc_id": doc.doc_id,
        "titre": doc.titre,
        "exercice": doc.exercice,
        "colonnes": [c[0] for c in columns],
        "libelles": {k: v for k, v in columns},
        "pas": min(steps) if steps else None,
        "pas_irreguliers": sorted(steps)[:10] if len(steps) > 1 else [],
        "lignes": ordered,
    }


def bareme_path(doc_id: str) -> Path:
    return get_settings().baremes_dir / f"{doc_id}.json"


def save_bareme(data: dict) -> Path:
    path = bareme_path(data["doc_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@dataclass
class BaremeResult:
    doc_id: str
    titre: str
    exercice: int | None
    montant_demande: int
    ligne_base: int | None
    valeurs: dict[str, int]
    libelles: dict[str, str]
    hors_bareme: bool
    minimum: int
    maximum: int

    def as_text(self) -> str:
        if self.hors_bareme:
            return (
                f"{self.titre} : le montant de {fcfa(self.montant_demande)} est hors du barème, "
                f"qui couvre de {fcfa(self.minimum)} à {fcfa(self.maximum)} par mois."
            )
        lines = [
            self.titre,
            f"Ligne du barème retenue : {fcfa(self.ligne_base)} (montant demandé : {fcfa(self.montant_demande)})",
        ]
        lines += [f"- {self.libelles[k]} : {fcfa(v)}" for k, v in self.valeurs.items() if k != "base"]
        return "\n".join(lines)


def fcfa(value: int) -> str:
    return f"{value:,}".replace(",", " ") + " FCFA"


class Bareme:
    def __init__(self, data: dict):
        self.data = data
        self.bases = [r[0] for r in data["lignes"]]

    @classmethod
    def load(cls, doc_id: str) -> "Bareme":
        return cls(json.loads(bareme_path(doc_id).read_text(encoding="utf-8")))

    def lookup(self, montant: float) -> BaremeResult:
        """Ligne dont la base est la plus grande valeur inférieure ou égale au montant."""
        montant = int(montant)
        d = self.data
        lo, hi = self.bases[0], self.bases[-1]
        common = dict(
            doc_id=d["doc_id"],
            titre=d["titre"],
            exercice=d.get("exercice"),
            montant_demande=montant,
            libelles=d["libelles"],
            minimum=lo,
            maximum=hi,
        )
        if montant < lo or montant > hi + (d.get("pas") or 0):
            return BaremeResult(ligne_base=None, valeurs={}, hors_bareme=True, **common)
        row = d["lignes"][bisect_right(self.bases, montant) - 1]
        return BaremeResult(
            ligne_base=row[0],
            valeurs=dict(zip(d["colonnes"], row, strict=True)),
            hors_bareme=False,
            **common,
        )


@lru_cache
def get_bareme(doc_id: str) -> Bareme | None:
    return Bareme.load(doc_id) if bareme_path(doc_id).exists() else None


def available_baremes() -> list[str]:
    return [doc_id for doc_id in BAREME_COLUMNS if bareme_path(doc_id).exists()]
