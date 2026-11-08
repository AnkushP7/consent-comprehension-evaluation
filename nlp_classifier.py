"""Supervised NLP Classifier for Consent Document Review.

Provides:
- Machine-learned sentence triage (clear vs needs_review).
- Subcategory classification (plain_language, jargon, legalistic, uncertainty, complexity).
- Transparent token-level explainability (which words drove the prediction).
- Document-level aggregated review indicators.
- Reproducible model training pipeline with GroupShuffleSplit by document.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "semantic_review_model.joblib"


class SemanticConsentClassifier:
    """Inference wrapper for the supervised consent review model."""

    _instance: Optional["SemanticConsentClassifier"] = None

    def __init__(self, model_path: Optional[str | Path] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.artifact: Optional[dict[str, Any]] = None
        self.pipeline = None
        self.issue_pipeline = None
        self.metadata: dict[str, Any] = {}
        self.classes_: list[str] = []
        self._load_model()

    @classmethod
    def get_default_instance(cls) -> "SemanticConsentClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            import joblib
            data = joblib.load(self.model_path)
            if isinstance(data, dict):
                self.artifact = data
                self.pipeline = data.get("pipeline")
                self.issue_pipeline = data.get("issue_pipeline")
                self.metadata = data.get("metadata", {})
            else:
                self.pipeline = data
                self.metadata = {}
            if self.pipeline and hasattr(self.pipeline, "classes_"):
                self.classes_ = [str(c) for c in self.pipeline.classes_]
            return True
        except Exception:
            self.artifact = None
            self.pipeline = None
            return False

    def is_available(self) -> bool:
        return self.pipeline is not None

    def explain_sentence(self, sentence: str, top_n: int = 5) -> list[dict[str, Any]]:
        """Identify which n-grams in the sentence contributed most to the review score.

        Uses local linear attribution: TF-IDF feature weight * model coefficient.
        Positive contribution increases 'needs_review' log-odds.
        """
        if not self.is_available() or not sentence or not sentence.strip():
            return []
        try:
            tfidf = self.pipeline.named_steps["tfidf"]
            classifier = self.pipeline.named_steps["classifier"]
            feature_names = np.array(tfidf.get_feature_names_out())

            # Find review class index (typically 'needs_review')
            if "needs_review" in self.classes_:
                review_idx = list(self.classes_).index("needs_review")
            else:
                review_idx = 1 if len(self.classes_) > 1 else 0

            coef = classifier.coef_[0] if len(classifier.coef_) == 1 else classifier.coef_[review_idx]

            # Transform single sentence
            vec = tfidf.transform([sentence])
            non_zero_indices = vec.indices
            if len(non_zero_indices) == 0:
                return []

            contributions = []
            for idx in non_zero_indices:
                feat_weight = vec[0, idx]
                c = coef[idx]
                attribution = float(feat_weight * c)
                contributions.append({
                    "token": str(feature_names[idx]),
                    "attribution": round(attribution, 4),
                    "direction": "needs_review" if attribution > 0 else "clear",
                })

            contributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)
            return contributions[:top_n]
        except Exception:
            return []

    def predict_sentence(self, sentence: str) -> dict[str, Any]:
        """Predict review necessity and issue type for a single sentence."""
        res = self.predict_sentences([sentence])
        return res[0] if res else {
            "sentence": sentence,
            "predicted_label": "unclassified",
            "review_probability": 0.0,
            "is_flagged": False,
            "predicted_issue_type": "none",
            "top_attributions": [],
        }

    def predict_sentences(self, sentences: list[str]) -> list[dict[str, Any]]:
        """Batch predict for a list of sentences."""
        if not self.is_available():
            return [{
                "sentence_number": idx + 1,
                "sentence": s,
                "predicted_label": "model_unavailable",
                "review_probability": 0.0,
                "confidence": 0.0,
                "is_flagged": False,
                "predicted_issue_type": "none",
                "top_attributions": [],
            } for idx, s in enumerate(sentences)]

        clean = [s if (s and s.strip()) else "empty" for s in sentences]
        try:
            preds = self.pipeline.predict(clean)
            probs = self.pipeline.predict_proba(clean) if hasattr(self.pipeline, "predict_proba") else None

            # Optional issue classifier predictions
            issue_preds = None
            if self.issue_pipeline is not None:
                try:
                    issue_preds = self.issue_pipeline.predict(clean)
                except Exception:
                    issue_preds = None

            results = []
            review_idx = list(self.classes_).index("needs_review") if "needs_review" in self.classes_ else 1

            for idx, orig_text in enumerate(sentences):
                pred_label = str(preds[idx])
                if probs is not None:
                    review_p = float(probs[idx][review_idx]) if review_idx < probs.shape[1] else float(max(probs[idx]))
                    conf = float(max(probs[idx]))
                else:
                    review_p = 1.0 if pred_label == "needs_review" else 0.0
                    conf = 1.0

                is_flagged = (pred_label == "needs_review") or (review_p >= 0.50)

                # Determine issue type (flagged sentences must never be plain_language)
                if not is_flagged:
                    issue_type = "plain_language"
                elif issue_preds is not None:
                    raw_issue = str(issue_preds[idx])
                    issue_type = "jargon" if raw_issue == "plain_language" else raw_issue
                else:
                    issue_type = "jargon"

                attributions = self.explain_sentence(orig_text, top_n=4) if is_flagged else []

                results.append({
                    "sentence_number": idx + 1,
                    "sentence": orig_text,
                    "predicted_label": pred_label,
                    "review_probability": round(review_p, 3),
                    "confidence": round(conf, 3),
                    "is_flagged": is_flagged,
                    "predicted_issue_type": issue_type,
                    "top_attributions": attributions,
                })
            return results
        except Exception as exc:
            return [{
                "sentence_number": idx + 1,
                "sentence": s,
                "predicted_label": "error",
                "review_probability": 0.0,
                "confidence": 0.0,
                "is_flagged": False,
                "predicted_issue_type": "none",
                "error": str(exc),
                "top_attributions": [],
            } for idx, s in enumerate(sentences)]

    def score_document(self, sentences: list[str]) -> dict[str, Any]:
        """Aggregate sentence-level classifier predictions into document-level metrics."""
        if not sentences:
            return {
                "classifier_active": self.is_available(),
                "total_sentences": 0,
                "flagged_count": 0,
                "flagged_share": 0.0,
                "mean_review_probability": 0.0,
                "max_review_probability": 0.0,
                "high_risk_sentences": [],
                "sentence_predictions": [],
                "issue_counts": {},
            }

        preds = self.predict_sentences(sentences)
        flagged = [p for p in preds if p["is_flagged"]]
        probs = [p["review_probability"] for p in preds]

        issue_counts: dict[str, int] = {}
        for p in flagged:
            it = p.get("predicted_issue_type", "general_review")
            issue_counts[it] = issue_counts.get(it, 0) + 1

        ranked_by_risk = sorted(flagged, key=lambda x: x["review_probability"], reverse=True)

        return {
            "classifier_active": self.is_available(),
            "model_type": self.metadata.get("model_type", "Supervised TF-IDF Logistic Regression"),
            "heldout_accuracy": self.metadata.get("accuracy"),
            "heldout_macro_f1": self.metadata.get("macro_f1"),
            "total_sentences": len(sentences),
            "flagged_count": len(flagged),
            "flagged_share": round(len(flagged) / len(sentences), 3) if sentences else 0.0,
            "mean_review_probability": round(float(np.mean(probs)), 3) if probs else 0.0,
            "max_review_probability": round(float(np.max(probs)), 3) if probs else 0.0,
            "issue_counts": issue_counts,
            "high_risk_sentences": ranked_by_risk[:5],
            "sentence_predictions": preds,
        }


def train_classifier(
    csv_path: str | Path,
    output_path: str | Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a supervised consent sentence classifier with GroupShuffleSplit."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.pipeline import Pipeline
    import joblib

    data = pd.read_csv(csv_path)
    required = {"text", "label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {', '.join(sorted(missing))}")

    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[(data["text"].str.len() > 0) & (data["label"].str.len() > 0)]

    counts = data["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("At least two distinct labels are required for training.")

    X = data["text"].to_numpy()
    y = data["label"].to_numpy()

    # Split with document grouping if available
    if "document_id" in data.columns and data["document_id"].nunique() >= 4:
        groups = data["document_id"].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        split_method = f"GroupShuffleSplit by document_id ({data['document_id'].nunique()} documents)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        split_method = "Stratified random row split"

    # Main binary classifier (clear vs needs_review)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=10000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("classifier", LogisticRegression(
            C=1.5,
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Subcategory classifier trained strictly on review-needing sentences
    issue_pipeline = None
    issue_metrics = {}
    if "issue_type" in data.columns:
        review_data = data[data["label"] == "needs_review"].copy()
        if review_data["issue_type"].nunique() >= 2:
            try:
                X_iss = review_data["text"].to_numpy()
                y_iss = review_data["issue_type"].astype(str).str.strip().to_numpy()
                issue_pipeline = Pipeline([
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000, sublinear_tf=True)),
                    ("classifier", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)),
                ])
                issue_pipeline.fit(X_iss, y_iss)
                issue_metrics = {
                    "issue_classes": sorted(set(y_iss)),
                }
            except Exception:
                issue_pipeline = None

    # Global feature importances (top predictive n-grams)
    top_features = {"needs_review": [], "clear": []}
    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["classifier"]
        feature_names = np.array(tfidf.get_feature_names_out())
        classes = list(clf.classes_)
        if "needs_review" in classes:
            r_idx = classes.index("needs_review")
            coef = clf.coef_[0] if len(clf.coef_) == 1 else clf.coef_[r_idx]
            top_review_idx = np.argsort(coef)[-15:][::-1]
            top_clear_idx = np.argsort(coef)[:15]
            top_features["needs_review"] = [
                {"feature": str(feature_names[i]), "weight": round(float(coef[i]), 4)} for i in top_review_idx
            ]
            top_features["clear"] = [
                {"feature": str(feature_names[i]), "weight": round(float(-coef[i]), 4)} for i in top_clear_idx
            ]
    except Exception:
        pass

    labels_sorted = sorted(set(y))
    metrics = {
        "model_type": "TF-IDF (1-2 grams) + LogisticRegression (balanced)",
        "dataset_path": str(Path(csv_path).resolve()),
        "rows_total": int(len(data)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "split_method": split_method,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist(),
        "classes": labels_sorted,
        "top_features": top_features,
        "issue_metrics": issue_metrics,
        "note": "Supervised NLP classifier for informed consent sentence review. Evaluated with held-out group test split.",
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "issue_pipeline": issue_pipeline, "metadata": metrics}, out_file)

    metrics_file = out_file.with_suffix(".metrics.json")
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
