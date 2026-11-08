import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pytest
except ImportError:
    pytest = None
from consent_analyzer import analyze_consent, calculate_readability, clean_text, simplify_sentence, split_sentences

SAMPLE = """Purpose: The purpose of this study is to evaluate a treatment.
Procedure: You will receive the medicine once each week.
Risks: Side effects include nausea and bleeding.
Benefits: The treatment may improve symptoms but there may be no direct benefit.
Alternatives: You may choose another treatment or no treatment.
Monitoring: Your health will be monitored during follow-up visits.
Voluntary participation: Taking part is voluntary. You may withdraw at any time without penalty and ask questions."""


def test_clean_text_normalizes_newlines_and_spaces():
    assert clean_text(" A  B\r\n\r\n\r\n C ") == "A B\n\n C"


def test_sentence_splitter_preserves_multiple_sentences():
    sentences = split_sentences("Purpose: First statement. Risks: Second statement. You may stop.")
    assert len(sentences) >= 3


def test_readability_returns_core_metrics():
    result = calculate_readability(SAMPLE)
    assert result["word_count"] > 0
    assert result["sentence_count"] >= 5
    assert result["flesch_kincaid_grade"] >= 0
    assert "gunning_fog_index" in result


def test_fully_populated_sample_detects_core_elements():
    result = analyze_consent(SAMPLE, target_grade=8)
    for key in ("purpose", "procedure", "risks", "benefits", "alternatives", "monitoring", "autonomy"):
        assert result["elements"][key]["present"] is True
    assert result["structural_completeness_score"] == 100.0


def test_conditional_elements_are_not_penalized_when_not_applicable():
    result = analyze_consent(SAMPLE)
    assert result["elements"]["reproductive"]["applicable"] is False
    assert result["elements"]["off_label"]["applicable"] is False
    assert "reproductive" not in result["missing_elements"]
    assert "off_label" not in result["missing_elements"]


def test_off_label_check_becomes_applicable_when_selected():
    result = analyze_consent(SAMPLE, procedure_type="Off-label drug use")
    assert result["elements"]["off_label"]["applicable"] is True
    assert "off_label" in result["missing_elements"]


def test_reproductive_check_becomes_applicable_when_selected():
    result = analyze_consent(SAMPLE, reproductive_applicable=True)
    assert result["elements"]["reproductive"]["applicable"] is True
    assert "reproductive" in result["missing_elements"]


def test_sri_is_bounded():
    result = analyze_consent("The investigational product may produce hemodynamic instability and myocardial ischemia. This is not fully understood. You hereby waive and indemnify. " * 5)
    assert 0 <= result["semantic_risk_index"] <= 100


def test_sri_exposes_components_and_disclaimer():
    result = analyze_consent(SAMPLE)
    assert result["sri_components"]
    assert "not a standardized clinical score" in result["methodology_note"]


def test_jargon_is_detected():
    result = analyze_consent("The patient will receive intravenous treatment. The pharmacokinetics are assessed.")
    found = {item["term"] for item in result["jargon"]}
    assert "intravenous" in found
    assert "pharmacokinetics" in found


def test_risk_language_is_categorized():
    result = analyze_consent("There are unknown risks. The clause says you waive all claims. Serious adverse events may occur.")
    categories = {item["category"] for item in result["risk_language"]}
    assert "Uncertainty / unknown outcomes" in categories
    assert "Legalistic / rights-related wording" in categories
    assert "Strong risk disclosure (review for clarity)" in categories


def test_serious_risk_disclosure_not_in_sri_uncertainty_penalty():
    only_disclosure = analyze_consent("Serious adverse events may occur. Some risks are life-threatening.")
    assert only_disclosure["strong_risk_disclosures"]
    assert only_disclosure["sri_components"]["Uncertainty / legalistic wording (max 15)"] == 0


def test_rewriter_does_not_claim_change_when_unchanged():
    result = simplify_sentence("Please discuss the risks with your doctor.")
    assert result["suggestion"] is None
    assert result["status"] == "Human rewrite recommended"


def test_rewriter_changes_known_jargon():
    result = simplify_sentence("The medicine is administered intravenously.")
    assert result["suggestion"]
    assert result["suggestion"] != "The medicine is administered intravenously."


def test_too_short_input_rejected():
    if pytest is not None:
        with pytest.raises(ValueError):
            analyze_consent("Yes, no.")
    else:
        try:
            analyze_consent("Yes, no.")
            assert False, "Expected ValueError"
        except ValueError:
            pass


def test_structural_score_is_valid_range():
    result = analyze_consent(SAMPLE, reproductive_applicable=True, off_label_applicable=True)
    assert 0 <= result["structural_completeness_score"] <= 100
    assert result["applicable_element_count"] == 9


def test_critical_missing_consent_items_raise_review_priority():
    result = analyze_consent("This is a short questionnaire study. The procedure uses three questions. Risks include mild discomfort.")
    assert result["semantic_risk_status"] == "RED"
    assert "alternatives" in result["missing_elements"]
    assert "autonomy" in result["missing_elements"]


def test_high_reading_grade_is_not_marked_green():
    text = "The investigational product demonstrates pharmacokinetic characteristics and immunogenicity. " * 8
    result = analyze_consent(text, target_grade=5.0)
    assert result["semantic_risk_status"] in {"YELLOW", "RED"}


def test_nlp_classifier_integration():
    result = analyze_consent(SAMPLE, use_nlp_classifier=True)
    assert "nlp_analysis" in result
    assert result["nlp_analysis"]["classifier_active"] is True
    assert "flagged_share" in result["nlp_analysis"]

    assert result["semantic_risk_status"] in {"YELLOW", "RED"}
