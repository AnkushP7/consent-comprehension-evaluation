# Self-test report

Checks completed on the enhanced Hybrid NLP Consent Comprehension Evaluation package:

- **Python syntax compilation:** `app.py`, `consent_analyzer.py`, `nlp_classifier.py`, and `train_semantic_classifier.py` compiled with zero syntax errors and zero warnings.
- **Unit test suite:** 26 automated tests passed (100% success rate):
  - 19 core analyzer tests covering text normalization, sentence segmentation, readability metrics (Flesch-Kincaid, Gunning Fog, ARI), structural completeness, conditional disclosures, medical jargon matching, legalistic risk language detection, conservative rewriter safeguards, SRI bounds, and priority escalations.
  - 7 NLP classifier tests covering model availability, `clear` vs `needs_review` predictions, subcategory issue prediction (e.g. `legalistic`, `jargon`), token-level linear attribution explainability ($w_i \cdot x_i$), edge-case handling, and document-level scoring.
- **Model training and verification:** `train_semantic_classifier.py` successfully trained and bundled `models/semantic_review_model.joblib` and `models/semantic_review_model.metrics.json` on the curated 120-sentence corpus (`data/annotated_consent_sentences.csv`) using `GroupShuffleSplit` across 12 distinct study contexts.
- **Explainability verification:** Token attribution verified on sample sentences; correctly highlighted exculpatory terms (`indemnify`, `hold harmless`, `waive`) and pharmacological terms (`pharmacokinetics`, `hemodynamic`, `ischemia`).
- **End-to-end smoke check:** All 6 synthetic demonstration documents (`SYN-001` through `SYN-006`) and `sample_consent.txt` loaded, analyzed, and generated audit CSV, summary CSV, and JSON outputs without errors.
- **Test execution:** Executable with either `python run_tests.py` (zero external test runner dependencies required) or `pytest -q`.
