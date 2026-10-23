from rag_fisc_v2.domain import DocType
from rag_fisc_v2.ingestion.chunking import (
    chunk_document,
    match_outline_heading,
    match_struct_heading,
)
from rag_fisc_v2.ingestion.cleaning import clean_pages
from rag_fisc_v2.ingestion.manifest import DocumentSpec
from rag_fisc_v2.ingestion.ocr import OcrPage


def _doc(doc_type: DocType, exercice: int | None = 2026) -> DocumentSpec:
    return DocumentSpec(
        doc_id=f"test-{doc_type.value}",
        fichier="x.pdf",
        type=doc_type,
        exercice=exercice,
        titre="Document de test",
    )


def _lines(*pages: str):
    return clean_pages([OcrPage(page=i + 1, markdown=p) for i, p in enumerate(pages)])


CGI_PAGE_1 = """# LIVRE PREMIER
IMPOTS ET TAXES

# TITRE PREMIER : IMPOT SUR LES SOCIETES

## CHAPITRE PREMIER : CHAMP D'APPLICATION

### SECTION I : PERSONNES IMPOSABLES

**Article 2.-** (1) Sont passibles de l'impôt sur les sociétés les sociétés anonymes, les sociétés
à responsabilité limitée et les sociétés en commandite par actions.

(2) Les dispositions de l'article 3 s'appliquent également aux établissements publics.

**Article 2 bis.-** Les personnes morales visées à l'article 2 ci-dessus sont tenues de déclarer
leur existence auprès du centre des impôts de rattachement dans les quinze jours.
"""

CGI_PAGE_2 = """## CHAPITRE II : DETERMINATION DE LA BASE IMPOSABLE

**Article 17.-** Le taux de l'impôt sur les sociétés est fixé à 30 %. Il est majoré de 10 % au titre
des centimes additionnels communaux, conformément aux articles 21, 87 et 92 bis du présent Code.

| Chiffre d'affaires | Taux |
|---|---|
| Inférieur ou égal à 3 milliards | 25 % |
| Supérieur à 3 milliards | 30 % |
"""


def test_struct_headings():
    assert match_struct_heading("# TITRE PREMIER : IMPOT SUR LES SOCIETES")[0] == 2
    assert match_struct_heading("SOUS-SECTION II : REVENUS")[0] == 6
    assert match_struct_heading("# **SOUS-TITRE II :**")[0] == 3
    assert match_struct_heading("# **LIVRE DEUXIEME**")[0] == 1
    assert match_struct_heading("### TITRE TROISIÈME")[0] == 2
    assert match_struct_heading("# CHAPITRE UNIQUE")[0] == 4
    assert match_struct_heading("PREMIERE PARTIE : IMPOTS D'ETAT")[0] == 0
    assert match_struct_heading("Section d'un texte ordinaire qui continue") is None


def test_outline_headings():
    assert match_outline_heading("I- DISPOSITIONS RELATIVES A L'IS")[0] == 2
    assert match_outline_heading("A. Impôt sur les sociétés")[0] == 3
    assert match_outline_heading("## Mesures nouvelles")[0] == 2


def test_cgi_one_chunk_per_article_with_hierarchy():
    chunks = chunk_document(_doc(DocType.CGI), _lines(CGI_PAGE_1, CGI_PAGE_2))
    articles = [c.metadata.get("article") for c in chunks]
    assert articles == ["2", "2 bis", "17"]

    art17 = chunks[2]
    assert "CHAPITRE II" in art17.metadata["hierarchie"]
    assert "TITRE PREMIER" in art17.metadata["hierarchie"]
    assert art17.metadata["page_debut"] == 2
    assert art17.metadata["articles_cites"] == ["21", "87", "92 bis"]
    # Le tableau reste entier dans le morceau de l'article.
    assert "| Supérieur à 3 milliards | 30 % |" in art17.text
    assert chunks[0].metadata["hierarchie"].startswith("LIVRE PREMIER : IMPOTS ET TAXES")


def test_long_article_is_split_with_repeated_header():
    para = "Les contribuables relevant du régime du réel sont tenus de souscrire une déclaration. " * 12
    page = "**Article 18.-** " + "\n\n".join([para] * 4)
    chunks = chunk_document(_doc(DocType.CGI), _lines(page), max_chars=1500)
    assert len(chunks) > 1
    assert all(c.metadata["article"] == "18" for c in chunks)
    assert chunks[1].text.startswith("Article 18 (suite)")
    assert all(len(c.text) <= 1500 for c in chunks)


def test_big_table_split_repeats_header():
    rows = "\n".join(f"| {i} 000 | {i} |" for i in range(400))
    page = f"**Article 69.-** Barème.\n\n| Base | Impôt |\n|---|---|\n{rows}"
    chunks = chunk_document(_doc(DocType.CGI), _lines(page), max_chars=1500)
    table_chunks = [c for c in chunks if "| Base | Impôt |" in c.text]
    assert len(table_chunks) > 1


LFI_PAGE = """LOI N° 2025/012 DU 17 DECEMBRE 2025 PORTANT LOI DE FINANCES POUR L'EXERCICE 2026

PREMIERE PARTIE : DISPOSITIONS FISCALES

ARTICLE PREMIER.- Les dispositions du Code Général des Impôts sont modifiées et complétées ainsi qu'il suit :

Article 17.- Le taux de l'impôt sur les sociétés est fixé à 28 % pour les entreprises
dont le chiffre d'affaires est inférieur ou égal à trois milliards.

Article 92 bis.- Les entreprises publiques sont tenues d'opérer la retenue à la source de la TVA
dans les conditions prévues par voie d'arrêté du ministre chargé des finances.

ARTICLE DEUXIEME.- Les droits de douane à l'importation du riz sont fixés à 5 % pour compter du
1er janvier 2026, en application de l'article 3 de la présente loi.
"""


def test_lfi_links_quoted_cgi_articles():
    chunks = chunk_document(_doc(DocType.LFI), _lines(LFI_PAGE))
    by_cgi = {c.metadata.get("articles_cites", [None])[0]: c for c in chunks}
    assert "17" in by_cgi and by_cgi["17"].metadata["article"] == "1"
    assert by_cgi["92 bis"].text.startswith("Article 92 bis")
    deux = [c for c in chunks if c.metadata.get("article") == "2"]
    assert deux and "articles_cites" not in deux[0].metadata  # renvoi à la loi, pas au CGI
    assert all("DISPOSITIONS FISCALES" in c.metadata["hierarchie"] for c in chunks if c.metadata.get("article"))


CIRC_PAGE = """# CIRCULAIRE PRECISANT LES MODALITES D'APPLICATION DE LA LOI DE FINANCES 2026

I- DISPOSITIONS RELATIVES A L'IMPOT SUR LES SOCIETES

A- Réduction du taux de l'IS pour les PME

La loi de finances pour l'exercice 2026 modifie l'article 17 du CGI. Le taux de l'IS est réduit à
28 % pour les entreprises dont le chiffre d'affaires n'excède pas trois milliards de francs CFA.

Exemple : une société dont le bénéfice fiscal est de 100 000 000 FCFA paie 28 000 000 FCFA d'IS,
majorés des centimes additionnels communaux.

B- Retenue à la source

Les articles 92 bis et 143 du CGI étendent la liste des entités tenues de retenir la TVA.
"""


def test_circulaire_sections_and_refs():
    chunks = chunk_document(_doc(DocType.CIRCULAIRE), _lines(CIRC_PAGE), min_chars=100)
    assert chunks
    refs = {r for c in chunks for r in c.metadata.get("articles_cites", [])}
    assert {"17", "92 bis", "143"} <= refs
    is_chunk = next(c for c in chunks if "17" in c.metadata.get("articles_cites", []))
    assert "I- DISPOSITIONS RELATIVES" in is_chunk.metadata["hierarchie"]


def test_running_headers_removed():
    pages = [
        OcrPage(page=i, markdown=f"Code Général des Impôts – Edition 2026\n\n**Article {i}.-** Texte {i}.\n\n{i}")
        for i in range(1, 11)
    ]
    lines = clean_pages(pages)
    texts = [line.text for line in lines if line.text.strip()]
    assert not any("Edition 2026" in t for t in texts)
    assert sum(1 for t in texts if t.startswith("**Article")) == 10


def test_long_article_suites_keep_internal_subtitles():
    para = "Les charges sont déductibles dans les conditions prévues par la réglementation en vigueur. " * 10
    page = (
        "**Article 7.-** Le bénéfice net imposable est établi sous déduction des charges.\n\n"
        "#### **A - Frais généraux**\n\n"
        f"{para}\n\n"
        "#### **1- Rémunérations et prestations diverses**\n\n"
        f"{para}\n\n{para}\n\n"
        "#### **B - Charges financières**\n\n"
        f"{para}\n\n{para}"
    )
    chunks = chunk_document(_doc(DocType.CGI), _lines(page), max_chars=1500)
    headers = [c.text.splitlines()[0] for c in chunks[1:]]
    assert any(h.endswith("A - Frais généraux > 1- Rémunérations et prestations diverses") for h in headers)
    assert headers[-1].endswith("— B - Charges financières")


APPLICATION_PAGES = """# LIVRE PREMIER : IMPOTS ET TAXES

# TITRE I : IMPOTS DIRECTS

**Article 5.-** Les bénéfices passibles de l'impôt sur les sociétés sont déterminés en tenant compte des bénéfices réalisés au Cameroun.

# **ANNEXE IV : LISTE DES ORGANISMES A BUT NON LUCRATIF**

Association des parents d'élèves du Centre, Yaoundé, habilitée à opérer la retenue à la source de la TVA.

# **DECISION N° 0000127/MINFI/DGI DU 12 FEVRIER 2016**

# ***LE MINISTRE DES FINANCES,***

**Article 5.** - Le Directeur Général des Impôts est chargé de l'application de la présente décision qui sera publiée.

### TITRE III : DES DISPOSITIONS FINALES

La présente décision abroge toutes les dispositions antérieures contraires et entre en vigueur dès sa signature.
"""


def test_application_texts_are_not_cgi_articles():
    chunks = chunk_document(_doc(DocType.CGI), _lines(APPLICATION_PAGES), min_chars=50)
    cgi_articles = [c for c in chunks if c.metadata.get("article")]
    assert [c.metadata["article"] for c in cgi_articles] == ["5"]
    assert cgi_articles[0].text.startswith("**Article 5.-** Les bénéfices")

    annex = next(c for c in chunks if "Association des parents" in c.text)
    assert annex.metadata["hierarchie"].endswith("ANNEXE IV : LISTE DES ORGANISMES A BUT NON LUCRATIF")
    assert "TITRE I : IMPOTS DIRECTS" in annex.metadata["hierarchie"]

    decision = next(c for c in chunks if "chargé de l'application" in c.text)
    assert "article" not in decision.metadata
    assert decision.metadata["hierarchie"].startswith("TEXTES D'APPLICATION > DECISION N° 0000127")

    final = next(c for c in chunks if "abroge" in c.text)
    assert final.metadata["hierarchie"].startswith("TEXTES D'APPLICATION > DECISION N° 0000127")
    assert "LIVRE PREMIER" not in final.metadata["hierarchie"]


def test_toc_lines_with_three_dots_are_removed():
    lines = _lines("SECTION II : OBLIGATIONS DES REDEVABLES...105\n\nANNEXE I : Liste des biens … 107\n\nTexte normal.")
    assert [line.text for line in lines if line.text.strip()] == ["Texte normal."]


ANNEX_ZONE_PAGES = """# LIVRE TROISIEME : FISCALITE LOCALE

**Article C 12.-** La patente est due par toute personne exerçant une activité commerciale au Cameroun.

# **ANNEXE 1**

**PARTIE LEGISLATIVE**

# **LIVRE PREMIER**

**Article 17.-** La présente loi sera enregistrée et publiée selon la procédure d'urgence au Journal Officiel.

# **ANNEXE 2 PARTIE REGLEMENTAIRE**

# **DECRET N° 2015/2517/PM DU 16 JUILLET 2015**

**Article 3.-** Le présent décret fixe les modalités d'application de certaines dispositions fiscales.
"""


def test_annex_zone_has_no_cgi_articles():
    chunks = chunk_document(_doc(DocType.CGI), _lines(ANNEX_ZONE_PAGES), min_chars=50)
    assert [c.metadata.get("article") for c in chunks if c.metadata.get("article")] == ["C12"]
    loi = next(c for c in chunks if "procédure d'urgence" in c.text)
    assert "article" not in loi.metadata
    assert loi.metadata["hierarchie"].startswith("ANNEXES DU CGI > PARTIE LEGISLATIVE")
    decret = next(c for c in chunks if "modalités d'application" in c.text)
    assert decret.metadata["hierarchie"].startswith("ANNEXES DU CGI > PARTIE REGLEMENTAIRE > DECRET N° 2015/2517")
