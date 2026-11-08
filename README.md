# NLP-Based Consent Form Comprehension Evaluation

A runnable, comprehensive Streamlit application for the project **“NLP-Based Consent Form Comprehension Evaluation: Design of an AI Readability Tool for Patient Informed Consent.”**

This build provides a **Hybrid AI Screening Pipeline** uniting deterministic readability formulas, structural checklist validation, a medical/legal jargon lexicon, and a **fully bundled Supervised NLP Classifier** with local explainability (XAI) and an interactive **NLP Classifier Studio**.

---

## Key Features

1. **Supervised NLP Sentence Classifier:**
   - Bundled model in `models/semantic_review_model.joblib` trained on 120 curated clinical consent sentences across 12 clinical domains (`data/annotated_consent_sentences.csv`).
   - Group-split validation (`GroupShuffleSplit` by document) to prevent cross-document vocabulary leakage.
   - Dual-task inference: classifies sentences as `clear` vs `needs_review` and predicts issue subcategories (`plain_language`, `jargon`, `legalistic`, `uncertainty`, `complexity`).
   - Token-level explainability: local linear attribution ($w_i \cdot x_i$) identifies the exact vocabulary n-grams that triggered the flag.

2. **Interactive NLP Classifier Studio (New Tab!):**
   - **Live Sandbox:** Type or select any consent sentence and get instantaneous ML classification, review probability gauge, and visual attribution badges for individual words.
   - **Global Model Diagnostics:** Inspect held-out accuracy, macro-F1, confusion matrix, and top predictive n-grams for both classes.
   - **Curated Dataset Explorer:** View and download the 120-sentence annotated training corpus.

3. **Readability & Lexical Screening:**
   - Flesch–Kincaid Grade Level, Gunning Fog Index, Automated Readability Index (ARI), and Flesch Reading Ease.
   - Expanded 50+ term clinical/legal jargon lexicon with patient-friendly alternatives.
   - Pattern screening for legalistic waivers, indemnification clauses, and ambiguous risk disclosures.

4. **Structural Completeness Checklist:**
   - Screens for 7 core consent elements: Purpose, Procedure, Risks, Benefits, Alternatives, Monitoring, and Autonomy.
   - Conditional disclosures: Reproductive / pregnancy warnings and Off-label drug use explanations.

5. **Document Ingestion & Multi-Format Exports:**
   - Ingests pasted text, `.txt`, text-based `.pdf`, and `.docx` files.
   - Batch audit tab: Upload a CSV with multiple consent documents for bulk review.
   - One-click downloads of Detailed Audit CSV, Summary Record CSV, and Full JSON.

---

## Quick Start on Windows

### 1. Launch App via Double-Click
Double-click `run_windows.bat`. It will configure the local virtual environment and launch Streamlit at `http://localhost:8501`.

### 2. Or Run from Command Line
```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Launch Streamlit app
python -m streamlit run app.py
```

### 3. Run Automated Tests
```powershell
python run_tests.py
```
Or if pytest is installed:
```powershell
pytest -q
```
All 26 automated unit tests run across readability, structural checks, jargon detection, rewriter rules, and the supervised NLP classifier.

### 4. Retrain the NLP Classifier (Optional)
```powershell
python train_semantic_classifier.py
```
Trains on `data/annotated_consent_sentences.csv` (or specify a custom CSV path) and outputs the updated model artifact and metrics JSON.

---

## Project Structure

- `app.py` — Streamlit web interface with Document Analysis, NLP Studio, Batch Audit, Synthetic Examples, and Methodology tabs.
- `nlp_classifier.py` — Standalone inference engine and trainer (`SemanticConsentClassifier`, local linear attribution, document scoring).
- `consent_analyzer.py` — Core hybrid analysis logic, readability formulas, structural checklist, lexicon, and SRI.
- `train_semantic_classifier.py` — Model training script with GroupShuffleSplit evaluation and metrics reporting.
- `run_tests.py` — Universal test runner discovering and running all tests.
- `data/annotated_consent_sentences.csv` — Curated 120-sentence annotated consent corpus across 12 study contexts.
- `data/synthetic_consent_examples.csv` — Six synthetic demonstration consent documents for UI and batch testing.
- `data/annotation_template.csv` — Starter schema for institutional annotation.
- `models/semantic_review_model.joblib` — Trained model bundle (TF-IDF + Logistic Regression).
- `models/semantic_review_model.metrics.json` — Evaluated metrics and top feature weights.
- `tests/test_consent_analyzer.py` — Unit tests for rule-based analysis and hybrid integration.
- `tests/test_nlp_classifier.py` — Unit tests for classifier inference, probabilities, and explainability.
- `sample_consent.txt` — Bundled sample consent document.
- `requirements.txt` — Core runtime dependencies.
- `requirements-ml.txt` — ML dependencies (`scikit-learn`, `joblib`).

---

## Ethical & Clinical Limitations

- **Not an individual comprehension predictor:** Readability formulas and NLP classifier review probabilities measure textual characteristics and disclosure signals; they do not predict an individual patient's cognitive comprehension or health literacy.
- **Decision support only:** All outputs must be evaluated by qualified clinicians, researchers, and Institutional Review Boards (IRBs).
- **Privacy:** Document processing runs strictly locally and does not transmit data to external APIs.
