"""Unit tests for the supervised NLP classifier module."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nlp_classifier import SemanticConsentClassifier, train_classifier


class TestSemanticConsentClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = SemanticConsentClassifier()

    def test_model_is_available(self):
        self.assertTrue(self.clf.is_available(), "Trained model should be available.")

    def test_clear_sentence_prediction(self):
        sentence = "Taking part in this study is voluntary and you may stop at any time."
        pred = self.clf.predict_sentence(sentence)
        self.assertEqual(pred["predicted_label"], "clear")
        self.assertLess(pred["review_probability"], 0.50)
        self.assertFalse(pred["is_flagged"])

    def test_jargon_sentence_prediction(self):
        sentence = "The investigational product exhibits complex pharmacokinetics and may induce acute myocardial ischemia."
        pred = self.clf.predict_sentence(sentence)
        self.assertEqual(pred["predicted_label"], "needs_review")
        self.assertGreater(pred["review_probability"], 0.50)
        self.assertTrue(pred["is_flagged"])

    def test_legalistic_sentence_prediction(self):
        sentence = "The undersigned participant hereby agrees to waive any and all claims and hold harmless the sponsor."
        pred = self.clf.predict_sentence(sentence)
        self.assertEqual(pred["predicted_label"], "needs_review")
        self.assertTrue(pred["is_flagged"])
        self.assertEqual(pred["predicted_issue_type"], "legalistic")

    def test_explain_sentence_returns_attributions(self):
        sentence = "You agree to indemnify and hold harmless the hospital notwithstanding severe adverse events."
        attributions = self.clf.explain_sentence(sentence, top_n=5)
        self.assertIsInstance(attributions, list)
        self.assertGreater(len(attributions), 0)
        top_tokens = [a["token"] for a in attributions]
        # At least one key term should appear in top attributions
        has_key_term = any(term in " ".join(top_tokens) for term in ["indemnify", "harmless", "hold", "adverse", "events", "severe"])
        self.assertTrue(has_key_term)

    def test_empty_and_short_inputs(self):
        pred_empty = self.clf.predict_sentence("")
        self.assertIsNotNone(pred_empty)
        pred_short = self.clf.predict_sentence("Hi.")
        self.assertIsNotNone(pred_short)

    def test_score_document_aggregation(self):
        sentences = [
            "This study asks you to complete one short survey.",
            "Taking part is voluntary and you may stop at any time.",
            "The participant hereby waives all claims and indemnifies the sponsor.",
            "Possible adverse events include hemodynamic instability and myocardial ischemia.",
        ]
        doc_score = self.clf.score_document(sentences)
        self.assertTrue(doc_score["classifier_active"])
        self.assertEqual(doc_score["total_sentences"], 4)
        self.assertGreaterEqual(doc_score["flagged_count"], 1)
        self.assertGreater(doc_score["flagged_share"], 0.0)
        self.assertGreater(doc_score["mean_review_probability"], 0.0)
        self.assertIn("sentence_predictions", doc_score)
        self.assertEqual(len(doc_score["sentence_predictions"]), 4)


if __name__ == "__main__":
    unittest.main()
