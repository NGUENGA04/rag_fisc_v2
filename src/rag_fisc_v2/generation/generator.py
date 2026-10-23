"""Génération de la réponse finale avec Mistral Medium."""

from dataclasses import dataclass

from rag_fisc_v2.config import get_settings
from rag_fisc_v2.domain import Canal, LlmUsage, Profil
from rag_fisc_v2.generation.citations import find_citations, normalize_citations
from rag_fisc_v2.generation.prompts import system_prompt, user_prompt
from rag_fisc_v2.ingestion.baremes import BaremeResult
from rag_fisc_v2.retrieval.retriever import Passage


@dataclass
class Generation:
    text: str
    cited: list[str]  # références effectivement citées, ex. ["S1", "S3", "B"]


def format_sources(passages: list[Passage], bareme: BaremeResult | None) -> tuple[str, dict[str, Passage]]:
    blocks: list[str] = []
    refs: dict[str, Passage] = {}
    if bareme is not None:
        blocks.append(f"[B] Résultat exact du barème\n{bareme.as_text()}")
    for i, p in enumerate(passages, start=1):
        ref = f"S{i}"
        refs[ref] = p
        m = p.metadata
        exercice = m.get("exercice") or "non daté"
        plan = f"\nPlan : {m['hierarchie']}" if m.get("hierarchie") else ""
        blocks.append(f"[{ref}] {p.label()} (exercice {exercice}){plan}\n{p.text}")
    return "\n\n---\n\n".join(blocks), refs


def generate_answer(
    question: str,
    profil: Profil,
    passages: list[Passage],
    exercices: list[int],
    warnings: list[str],
    bareme: BaremeResult | None = None,
    canal: Canal = Canal.API,
    question_complete: str | None = None,
    usage: LlmUsage | None = None,
) -> tuple[Generation, dict[str, Passage]]:
    from rag_fisc_v2.clients import get_mistral

    settings = get_settings()
    sources_block, refs = format_sources(passages, bareme)
    response = get_mistral().chat.complete(
        model=settings.generation_model,
        temperature=0.1,
        max_tokens=settings.generation_max_tokens,
        messages=[
            {"role": "system", "content": system_prompt(profil, canal)},
            {
                "role": "user",
                "content": user_prompt(question, exercices, sources_block, warnings, question_complete),
            },
        ],
    )
    if usage is not None:
        usage.add(response.usage)
    content = response.choices[0].message.content
    if isinstance(content, list):  # certains modèles renvoient des blocs
        content = "".join(getattr(c, "text", "") for c in content)
    text = normalize_citations((content or "").strip())
    return Generation(text=text, cited=find_citations(text)), refs
