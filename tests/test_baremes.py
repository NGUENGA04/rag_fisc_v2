from rag_fisc_v2.ingestion.baremes import Bareme, fcfa

DATA = {
    "doc_id": "bareme-test",
    "titre": "Barème de test",
    "exercice": None,
    "colonnes": ["base", "irpp"],
    "libelles": {"base": "Base", "irpp": "IRPP"},
    "pas": 250,
    "lignes": [[52000, 0], [52250, 10], [52500, 20]],
}


def test_lookup_takes_lower_bracket():
    r = Bareme(DATA).lookup(52400)
    assert not r.hors_bareme
    assert r.ligne_base == 52250
    assert r.valeurs["irpp"] == 10


def test_lookup_exact_and_last_step():
    b = Bareme(DATA)
    assert b.lookup(52500).valeurs["irpp"] == 20
    assert b.lookup(52700).valeurs["irpp"] == 20  # dans le dernier pas
    assert b.lookup(52800).hors_bareme


def test_lookup_below_range():
    r = Bareme(DATA).lookup(40000)
    assert r.hors_bareme
    assert "hors du barème" in r.as_text()


def test_fcfa_format():
    assert fcfa(1500000) == "1 500 000 FCFA"
