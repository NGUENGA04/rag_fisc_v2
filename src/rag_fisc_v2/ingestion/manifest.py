"""Lecture et validation du manifeste des documents (data/manifest.yaml)."""

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from rag_fisc_v2.config import get_settings
from rag_fisc_v2.domain import DOC_TYPE_LABELS, EXERCICE_NON_DATE, DocType


class DocumentSpec(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]*$")
    fichier: str
    type: DocType
    exercice: int | None = None
    titre: str
    reference: str | None = None
    ocr: bool = True
    indexer: bool = True

    @field_validator("exercice")
    @classmethod
    def _exercice_plausible(cls, v: int | None) -> int | None:
        if v is not None and not 1990 <= v <= 2100:
            raise ValueError(f"exercice invalide : {v}")
        return v

    @property
    def exercice_meta(self) -> int:
        """Valeur stockée dans Pinecone, qui n'accepte pas null."""
        return self.exercice if self.exercice is not None else EXERCICE_NON_DATE

    @property
    def type_label(self) -> str:
        return DOC_TYPE_LABELS[self.type]

    def path(self, raw_dir: Path | None = None) -> Path:
        return (raw_dir or get_settings().raw_dir) / self.fichier

    def sha256(self, raw_dir: Path | None = None) -> str:
        h = hashlib.sha256()
        with self.path(raw_dir).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()


class Manifest(BaseModel):
    documents: list[DocumentSpec]

    @field_validator("documents")
    @classmethod
    def _ids_uniques(cls, docs: list[DocumentSpec]) -> list[DocumentSpec]:
        seen: set[str] = set()
        for d in docs:
            if d.doc_id in seen:
                raise ValueError(f"doc_id en double : {d.doc_id}")
            seen.add(d.doc_id)
        return docs

    def get(self, doc_id: str) -> DocumentSpec:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        raise KeyError(f"doc_id inconnu : {doc_id}")

    def select(self, doc_ids: list[str] | None = None) -> list[DocumentSpec]:
        if not doc_ids:
            return list(self.documents)
        return [self.get(i) for i in doc_ids]

    def exercices(self, doc_type: DocType | None = None) -> list[int]:
        return sorted(
            {
                d.exercice
                for d in self.documents
                if d.exercice is not None and (doc_type is None or d.type == doc_type)
            }
        )

    def latest_exercice(self) -> int:
        years = self.exercices()
        if not years:
            raise ValueError("Aucun document daté dans le manifeste.")
        return years[-1]

    def cgi_for(self, exercice: int) -> DocumentSpec | None:
        for d in self.documents:
            if d.type == DocType.CGI and d.exercice == exercice:
                return d
        return None

    def latest_cgi(self) -> DocumentSpec | None:
        cgis = [d for d in self.documents if d.type == DocType.CGI and d.exercice is not None]
        return max(cgis, key=lambda d: d.exercice) if cgis else None

    def missing_files(self, raw_dir: Path | None = None) -> list[DocumentSpec]:
        return [d for d in self.documents if not d.path(raw_dir).exists()]


def load_manifest(path: Path | None = None) -> Manifest:
    path = path or get_settings().manifest_path
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Manifest.model_validate(data)
