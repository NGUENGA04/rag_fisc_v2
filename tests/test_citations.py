import pytest

from rag_fisc_v2.generation.citations import find_citations, normalize_citations


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Taux de 30 % [S1, art. 17(1)].", "Taux de 30 % [S1]."),
        ("Voir [S1, S3] et [S2; B].", "Voir [S1][S3] et [S2][B]."),
        ("Déjà correct [S1][S4].", "Déjà correct [S1][S4]."),
        ("Un crochet ordinaire [voir annexe B] reste.", "Un crochet ordinaire [voir annexe B] reste."),
        ("[S12 ; al. 2]", "[S12]"),
    ],
)
def test_normalize(raw, expected):
    assert normalize_citations(raw) == expected


def test_find_citations_order_and_unique():
    assert find_citations("A [S3]. B [S1, S3]. C [B]. D [annexe S9 ?]") == ["S3", "S1", "B"]
