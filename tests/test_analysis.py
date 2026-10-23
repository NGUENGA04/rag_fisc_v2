import pytest

from rag_fisc_v2.retrieval.analysis import (
    QueryAnalysis,
    extract_amount,
    merge_analyses,
    rule_based_analysis,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("salaire de 350 000 FCFA", 350000),
        ("un salaire de 350000 F", 350000),
        ("1,5 million de francs", 1_500_000),
        ("2 millions", 2_000_000),
        ("en 2025, quel taux ?", None),
        ("article 7", None),
    ],
)
def test_extract_amount(text, expected):
    assert extract_amount(text) == expected


def test_rules_detect_bareme_calculation():
    a = rule_based_analysis("Combien d'IRPP sur un salaire de 350 000 FCFA ?")
    assert a.intention == "calcul_bareme"
    assert a.bareme == "bareme-irpp"
    assert a.montant == 350000


def test_rules_detect_years_and_articles():
    a = rule_based_analysis("Que prévoit l'article 92 bis du CGI en 2025 ?")
    assert a.exercices == [2025]
    assert a.articles == ["92 bis"]
    assert a.intention == "information"


def test_rules_detect_comparison():
    assert rule_based_analysis("Taux de TVA en 2024 et 2026").intention == "comparaison"


def test_merge_rejects_invented_years_and_articles():
    rules = rule_based_analysis("Quel est le taux de l'IS ?")
    llm = QueryAnalysis(
        question_reformulee="Quel est le taux de l'impôt sur les sociétés ?",
        intention="information",
        exercices=[2023],  # absent de la question : ignoré
        articles=["Article 17"],  # proposé de mémoire : ignoré
        impots=["IS"],
    )
    merged = merge_analyses(rules, llm)
    assert merged.exercices == []
    assert merged.articles == []
    assert merged.question_reformulee.startswith("Quel est le taux de l'impôt")


def test_merge_normalizes_articles_present_in_question():
    rules = rule_based_analysis("Que dit l'art 92 BIS ?")  # forme que les règles ne captent pas
    llm = QueryAnalysis(question_reformulee="x", intention="information", articles=["Article 92 BIS"])
    assert merge_analyses(rules, llm).articles == ["92 bis"]


def test_merge_downgrades_calculation_without_amount():
    rules = rule_based_analysis("Comment est calculé l'IRPP ?")
    llm = QueryAnalysis(question_reformulee="x", intention="calcul_bareme", bareme="bareme-irpp")
    assert merge_analyses(rules, llm).intention == "information"
