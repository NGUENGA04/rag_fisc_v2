import pytest

from rag_fisc_v2.ingestion.references import (
    extract_article_refs,
    match_cgi_article_header,
    match_ordinal_article_header,
    parse_article_id,
    parse_french_ordinal,
)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Article 7.- (1) Sont imposables", "7"),
        ("**Article 92 bis.-** Les", "92 bis"),
        ("ARTICLE 92 BIS : texte", "92 bis"),
        ("# Art. L106 - Intérêts de retard", "L106"),
        ("Article C 12.-", "C12"),
        ("Article premier.-", "1"),
        ("Article 1er :", "1"),
        ("Article 15", "15"),
        ("L'article 7 du présent code", None),
        ("Article 7 du CGI dispose que", None),
    ],
)
def test_cgi_article_header(line, expected):
    assert match_cgi_article_header(line) == expected


@pytest.mark.parametrize(
    "line,expected",
    [
        ("ARTICLE HUITIÈME.- Relèvement graduel", 8),
        ("Article premier.- Les dispositions", 1),
        ("ARTICLE VINGT-ET-UNIÈME.-", 21),
        ("ARTICLE SOIXANTE-DOUZIEME :", 72),
        ("ARTICLE QUATRE-VINGT-DIXIÈME.-", 90),
        ("ARTICLE TROISIEME", 3),
        ("Article 7.- texte", None),
    ],
)
def test_ordinal_article_header(line, expected):
    assert match_ordinal_article_header(line) == expected


def test_parse_french_ordinal_rejects_plain_words():
    assert parse_french_ordinal("dix") is None
    assert parse_french_ordinal("trente-deuxième") == 32


def test_extract_refs_lists_and_exclusions():
    text = (
        "En application des articles 21, 87, 92, 92 bis, 143 et 149 du Code Général des Impôts "
        "et de l'article L106. Voir l'article 3 de la présente loi. Article 6 (1) du CGI."
    )
    assert extract_article_refs(text) == ["21", "87", "92", "92 bis", "143", "149", "L106", "6"]


def test_header_with_comma_ocr_artifact():
    assert match_cgi_article_header("Article 5,- Toute compensation") == "5"


def test_refs_with_alinea_and_internal_refs():
    text = "En application des dispositions des articles 22(3), 81, 88 et 149 du Code Général des Impôts."
    assert extract_article_refs(text) == ["22", "81", "88", "149"]
    assert extract_article_refs("visés à l'article 10 du présent arrêté") == []
    assert extract_article_refs("prévue à l'article 8 alinéa 2 du CGI") == ["8"]


def test_header_line_is_not_a_reference():
    text = "Article 9.- Les entreprises visées à l'article 92 bis du CGI.\nArticle 10.- Texte."
    assert extract_article_refs(text) == ["92 bis"]


def test_parse_article_id():
    assert parse_article_id("92 BIS") == "92 bis"
    assert parse_article_id("L 106") == "L106"
    assert parse_article_id("bonjour") is None
