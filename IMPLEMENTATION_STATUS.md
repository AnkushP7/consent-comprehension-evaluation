# Implementation status against the project report

| Report component | Status in this build | Implementation detail / caveat |
|---|---|---|
| Consent text input and metadata | Implemented | Paste text, TXT, text-based PDF, DOCX; procedure type and target grade; applicability settings. |
| Preprocessing | Baseline implemented | Whitespace normalization and sentence segmentation. No advanced language detection or lemmatization. |
| Flesch–Kincaid grade | Implemented | `textstat` with a transparent local fallback. English-only assumption. |
| Gunning Fog / Flesch Reading Ease / ARI | Implemented | Automated Readability Index (ARI) added alongside Gunning Fog and Flesch Reading Ease with transparent formula fallbacks. |
| Structural completeness | Implemented as a rule baseline | Required elements use keyword/pattern cues. No legal/clinical adequacy judgment. Conditional checks for reproductive/off-label disclosures. |
| Semantic / risk flags | Implemented as a rule baseline | Expanded jargon lexicon (50+ terms), uncertainty/legalistic phrases, and serious-risk disclosure cues. |
| Semantic Risk Index | Implemented as custom heuristic | Individual contributions shown in the UI; not standardized or validated. |
| Plain-language rewriting | Implemented conservatively | Dictionary and phrase-level substitutions with IRB exculpatory alerts; unverified rewrites sent to clinical review. |
| Dashboard and exports | Implemented | Traffic-light review priority, metrics, evidence table, audit CSV, summary CSV, JSON. |
| Batch evaluation | Implemented | CSV with one consent document per row; includes NLP model flag rate and mean review probability. |
| Supervised NLP classifier | **Fully implemented & bundled** | TF-IDF (1-2 grams) + balanced Logistic Regression trained on curated 120-sentence consent corpus (`data/annotated_consent_sentences.csv`) with `GroupShuffleSplit` across 12 clinical study contexts. Includes issue subtyping (jargon, legalistic, uncertainty, complexity) and local token attribution ($w_i \cdot x_i$) explainability in a dedicated interactive Studio tab. |
| Word2Vec / transformer fine-tuning | Not bundled | Lightweight classical ML selected for zero-latency, CPU-friendly inference and transparent linear attribution without requiring GPU or heavyweight weights. |
| RAG over policies / standards | Not configured | Requires actual approved policy documents and jurisdiction-specific source selection. |
| Patient comprehension prediction | Not implemented | Requires ethically approved patient outcome data and validation. SRI and ML scores must not be presented as a probability of misunderstanding. |
| Clinical / legal validation and fairness tests | Not performed | Out of scope for a code-only prototype; requires qualified reviewers and an appropriate prospective study. |

This package provides a runnable, high-performance implementation of the **hybrid AI screening system** (transparent rules + supervised machine learning + explainability studio). It does not claim clinical or legal validation.
