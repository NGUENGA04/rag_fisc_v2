"""Modèles d'entrée et de sortie de l'API."""

from pydantic import BaseModel, Field

from rag_fisc_v2.domain import DocType, Profil


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["Quel est le taux de l'impôt sur les sociétés ?"])
    profil: Profil = Profil.GRAND_PUBLIC
    exercice: int | None = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Force l'exercice fiscal. Sinon, il est déduit de la question, ou le plus récent est utilisé.",
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=128,
        description="Identifiant de conversation fourni par le client. Active la mémoire des échanges "
        "(questions de suivi comme « et pour 2025 ? »).",
    )
    debug: bool = Field(default=False, description="Inclut l'analyse de la question dans la réponse.")


class Source(BaseModel):
    ref: str = Field(description="Référence utilisée dans la réponse, ex. S1")
    doc_id: str
    type: DocType
    titre_document: str
    exercice: int | None
    article: str | None = None
    articles_cites: list[str] = []
    hierarchie: str | None = None
    page_debut: int | None = None
    page_fin: int | None = None
    extrait: str
    score: float
    citee: bool = Field(description="Vrai si la réponse cite explicitement cette source.")


class CalculBareme(BaseModel):
    bareme: str
    titre: str
    montant_demande: int
    ligne_base: int | None
    valeurs: dict[str, int]
    libelles: dict[str, str]
    hors_bareme: bool


class AskResponse(BaseModel):
    reponse: str
    profil: Profil
    exercices: list[int]
    sources: list[Source]
    calcul: CalculBareme | None = None
    avertissements: list[str] = []
    analyse: dict | None = None
    interaction_id: int | None = Field(default=None, description="À renvoyer à /feedback pour noter la réponse.")


class FeedbackRequest(BaseModel):
    interaction_id: int
    utile: bool


class BaremeResponse(CalculBareme):
    texte: str


class DocumentInfo(BaseModel):
    doc_id: str
    titre: str
    type: DocType
    exercice: int | None
    reference: str | None
    fichier_present: bool
    indexe: bool
    morceaux: int | None
    indexe_le: str | None


class HealthResponse(BaseModel):
    statut: str
    mistral_configure: bool
    pinecone_configure: bool
    whatsapp_configure: bool
    stockage: str = Field(description="postgres, sqlite, ou erreur si la base est inaccessible")
    documents_indexes: int
    baremes_disponibles: list[str]
