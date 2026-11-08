# Model and dataset notes

## What is included

- `data/annotated_consent_sentences.csv`: A curated and balanced corpus of 120 informed consent sentences spanning 12 distinct clinical research domains (observational, oncology phase 1, chronic pain, cardiology devices, pediatric asthma, surgical, metabolic, genetics/biobanking, maternal health, vaccines, and neurology). Each sentence is labelled with:
  - `label`: `clear` or `needs_review`
  - `issue_type`: `plain_language`, `jargon`, `legalistic`, `uncertainty`, or `complexity`
  - `document_id`: `DOC-001` through `DOC-012` for multi-document leakage prevention during cross-validation
- `models/semantic_review_model.joblib`: Pre-trained and bundled supervised NLP classification pipeline (TF-IDF unigrams and bigrams + sublinear term-frequency scaling + class-balanced Logistic Regression) paired with a subcategory issue classifier.
- `models/semantic_review_model.metrics.json`: Performance metrics, confusion matrix, and top predictive n-gram weights.
- `nlp_classifier.py`: Standalone inference and training module providing `SemanticConsentClassifier` with token-level explainability (local linear attribution $w_i \cdot x_i$).
- `data/synthetic_consent_examples.csv`: Six invented consent-like texts for testing end-to-end dashboard behavior across varying complexity profiles.
- `data/annotation_template.csv`: Annotation starter template for future IRB/clinical annotation efforts.
- `train_semantic_classifier.py`: Trainer and evaluation script supporting custom or bundled datasets with GroupShuffleSplit.

## Public sources for research corpus expansion

1. **UK cancer-trial participant information sheets and consent forms (2007–2017)** — King's College London / UK Data Service. Contains 27 participant information sheets and 23 consent forms as text files. Permitted-use terms: https://reshare.ukdataservice.ac.uk/853933/ . Dataset DOI: `10.5255/UKDA-SN-853933`.
2. **Spence et al. clinical trial ICF dataset** — includes ICFs for 98 industry-funded and 46 NIH-funded trials: https://zenodo.org/records/1231072 .
3. **AI-edited informed consent corpus (2025)** — includes original ICFs and AI-edited versions plus readability analysis: https://zenodo.org/records/15240810 .
4. **ClinicalTrials.gov** — public trial records with consent-form attachments: https://clinicaltrials.gov/ .

## Annotation protocol & group-level splitting

1. **Labels:** Binary `clear` vs `needs_review` paired with specific subcategories (`jargon`, `legalistic`, `uncertainty`, `complexity`).
2. **Leakage prevention:** Keep `document_id` in the dataset and use `GroupShuffleSplit`. Splitting random sentences from the same consent document creates severe data leakage because document-specific vocabulary (e.g., drug names) inflates test scores.
3. **Evaluation:** Track held-out accuracy, macro-F1, per-class precision/recall, and inspect top learned positive and negative feature coefficients.

## Retraining the model

To train on your own annotated CSV or retrain on the included corpus:

```bash
python train_semantic_classifier.py
```

Or with a custom dataset:

```bash
python train_semantic_classifier.py path/to/your_annotated_sentences.csv --output models/semantic_review_model.joblib
```

The app automatically loads `models/semantic_review_model.joblib` and exposes its predictions in both the **Analyze Document** dashboard and the **NLP Classifier Studio** tab.
