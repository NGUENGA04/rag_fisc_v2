"""Service principal : de la question à la réponse sourcée.

Le service ne dépend d'aucun canal. L'API web et WhatsApp l'appellent de la
même façon, avec un identifiant de conversation pour la mémoire des échanges.
"""

import logging
import time

from rag_fisc_v2 import storage
from rag_fisc_v2.domain import Canal, LlmUsage, Profil
from rag_fisc_v2.generation.generator import generate_answer
from rag_fisc_v2.generation.prompts import DISCLAIMER_GRAND_PUBLIC, NO_SOURCE_ANSWER, OFF_TOPIC_ANSWER
from rag_fisc_v2.ingestion.baremes import BaremeResult, get_bareme
from rag_fisc_v2.ingestion.manifest import Manifest
from rag_fisc_v2.retrieval.analysis import QueryAnalysis, analyze_question
from rag_fisc_v2.retrieval.retriever import FiscalRetriever, Passage
from rag_fisc_v2.schemas.api import AskRequest, AskResponse, CalculBareme, Source

log = logging.getLogger(__name__)

BAREME_YEAR_WARNING = (
    "Le barème utilisé ne précise pas son exercice d'application. Vérifiez qu'il correspond à la période concernée."
)


def _to_source(ref: str, p: Passage, cited: bool) -> Source:
    m = p.metadata
    return Source(
        ref=ref,
        doc_id=m.get("doc_id", ""),
        type=p.doc_type,
        titre_document=m.get("titre_document", ""),
        exercice=m.get("exercice") or None,
        article=m.get("article"),
        articles_cites=m.get("articles_cites", []),
        hierarchie=m.get("hierarchie") or None,
        page_debut=m.get("page_debut"),
        page_fin=m.get("page_fin"),
        extrait=p.text[:400] + ("…" if len(p.text) > 400 else ""),
        score=round(p.score, 4),
        citee=cited,
    )


def _to_calcul(result: BaremeResult) -> CalculBareme:
    return CalculBareme(
        bareme=result.doc_id,
        titre=result.titre,
        montant_demande=result.montant_demande,
        ligne_base=result.ligne_base,
        valeurs=result.valeurs,
        libelles=result.libelles,
        hors_bareme=result.hors_bareme,
    )


class RagService:
    def __init__(self, manifest: Manifest, retriever: FiscalRetriever | None = None, track: bool = True):
        self.manifest = manifest
        self.retriever = retriever or FiscalRetriever(manifest)
        self.track = track  # mémoire de conversation et suivi d'usage en base

    def bareme_for(self, analysis: QueryAnalysis) -> BaremeResult | None:
        if analysis.intention != "calcul_bareme" or analysis.bareme is None or analysis.montant is None:
            return None
        bareme = get_bareme(analysis.bareme)
        return bareme.lookup(analysis.montant) if bareme else None

    def ask(self, request: AskRequest, canal: Canal = Canal.API) -> AskResponse:
        started = time.monotonic()
        user_hash = storage.hash_user(f"{canal.value}:{request.conversation_id}") if request.conversation_id else None
        history = storage.get_history(user_hash) if (self.track and user_hash) else []
        usage = LlmUsage()
        log_fields: dict = {"canal": canal.value, "user_hash": user_hash or "anonyme", "profil": request.profil.value}

        try:
            response, analysis, reranker = self._answer(request, canal, history, usage)
        except Exception as exc:
            if self.track:
                storage.log_interaction(
                    **log_fields,
                    question=request.question,
                    erreur=f"{type(exc).__name__}: {exc}"[:500],
                    latence_ms=int((time.monotonic() - started) * 1000),
                )
            raise

        if self.track:
            if user_hash:
                storage.add_turn(user_hash, request.question, response.reponse, analysis.question_reformulee)
            response.interaction_id = storage.log_interaction(
                **log_fields,
                question=request.question,
                intention=analysis.intention,
                exercices=response.exercices,
                nb_sources=sum(1 for s in response.sources if s.citee),
                reranker=reranker,
                latence_ms=int((time.monotonic() - started) * 1000),
                tokens_entree=usage.prompt_tokens,
                tokens_sortie=usage.completion_tokens,
            )
        return response

    def _answer(
        self, request: AskRequest, canal: Canal, history: list, usage: LlmUsage
    ) -> tuple[AskResponse, QueryAnalysis, str]:
        analysis = analyze_question(request.question, history=history, usage=usage)
        if request.exercice:
            analysis.exercices = [request.exercice]
        debug = analysis.model_dump() if request.debug else None

        warnings: list[str] = []
        if request.profil == Profil.GRAND_PUBLIC:
            warnings.append(DISCLAIMER_GRAND_PUBLIC)

        if analysis.intention == "hors_sujet":
            return (
                AskResponse(reponse=OFF_TOPIC_ANSWER, profil=request.profil, exercices=[], sources=[], analyse=debug),
                analysis,
                "aucun",
            )

        bareme = self.bareme_for(analysis)
        if bareme is not None and bareme.exercice is None:
            warnings.append(BAREME_YEAR_WARNING)

        retrieval = self.retriever.retrieve(analysis)
        warnings = retrieval.avertissements + warnings

        if not retrieval.passages and bareme is None:
            return (
                AskResponse(
                    reponse=NO_SOURCE_ANSWER,
                    profil=request.profil,
                    exercices=retrieval.exercices,
                    sources=[],
                    avertissements=warnings,
                    analyse=debug,
                ),
                analysis,
                retrieval.reranker,
            )

        generation, refs = generate_answer(
            question=request.question,
            profil=request.profil,
            passages=retrieval.passages,
            exercices=retrieval.exercices,
            warnings=retrieval.avertissements,
            bareme=bareme,
            canal=canal,
            question_complete=analysis.question_reformulee,
            usage=usage,
        )
        sources = [_to_source(ref, p, ref in generation.cited) for ref, p in refs.items()]
        # Les sources citées d'abord, dans l'ordre de citation.
        order = {ref: i for i, ref in enumerate(generation.cited)}
        sources.sort(key=lambda s: (not s.citee, order.get(s.ref, 0)))

        response = AskResponse(
            reponse=generation.text,
            profil=request.profil,
            exercices=retrieval.exercices,
            sources=sources,
            calcul=_to_calcul(bareme) if bareme else None,
            avertissements=warnings,
            analyse=debug,
        )
        return response, analysis, retrieval.reranker
