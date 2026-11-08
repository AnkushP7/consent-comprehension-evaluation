# -*- coding: utf-8 -*-
"""Modern Streamlit interface for Consent Form Comprehension Evaluation & NLP Classifier."""
from __future__ import annotations

import html
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from consent_analyzer import analyze_consent, split_sentences, summary_record, words
from nlp_classifier import SemanticConsentClassifier

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    from docx import Document
except ImportError:
    Document = None

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_consent_examples.csv"
ANNOTATED_PATH = ROOT / "data" / "annotated_consent_sentences.csv"
SAMPLE_PATH = ROOT / "sample_consent.txt"
MODEL_METRICS_PATH = ROOT / "models" / "semantic_review_model.metrics.json"

st.set_page_config(
    page_title="Consent Form Comprehension Evaluation · Hybrid AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_styles():
    """Inject modern, clean medical-dashboard styling."""
    st.markdown("""
    <style>
    /* Global polish */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Header */
    .hero-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        color: #FFFFFF;
        padding: 24px 28px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #CBD5E1;
        margin: 0 0 14px 0;
        line-height: 1.5;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #F8FAFC;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
    }
    .kpi-icon {
        font-size: 22px;
        margin-bottom: 6px;
    }
    .kpi-label {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        margin-bottom: 6px !important;
    }
    .kpi-value {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        line-height: 1.15 !important;
        margin-bottom: 6px !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
    }
    .kpi-sub {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #CBD5E1 !important;
        margin-top: 4px;
    }

    /* Priority Triage Banners */
    .triage-banner {
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 14px;
        line-height: 1.5;
        border: 1px solid transparent;
    }
    .triage-red {
        background: #FEF2F2;
        border-color: #FCA5A5;
        color: #991B1B;
    }
    [data-theme="dark"] .triage-red {
        background: rgba(239, 68, 68, 0.15);
        border-color: rgba(239, 68, 68, 0.4);
        color: #FCA5A5;
    }
    .triage-yellow {
        background: #FFFBEB;
        border-color: #FDE68A;
        color: #92400E;
    }
    [data-theme="dark"] .triage-yellow {
        background: rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.4);
        color: #FDE68A;
    }
    .triage-green {
        background: #ECFDF5;
        border-color: #A7F3D0;
        color: #065F46;
    }
    [data-theme="dark"] .triage-green {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #6EE7B7;
    }

    /* Sentence Cards & Token Chips */
    .sentence-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .sentence-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .badge-review {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-clear {
        background: #D1FAE5;
        color: #047857;
        border: 1px solid #6EE7B7;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .token-chip {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 500;
        margin: 2px 4px 2px 0;
    }
    .token-red {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #DC2626;
    }
    [data-theme="dark"] .token-red {
        color: #F87171;
    }
    .token-green {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #059669;
    }
    [data-theme="dark"] .token-green {
        color: #34D399;
    }

    /* Comparison Box */
    .compare-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-top: 10px;
    }
    .compare-box {
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 13px;
        line-height: 1.45;
    }
    .compare-orig {
        background: rgba(148, 163, 184, 0.1);
        border-left: 3px solid #64748B;
    }
    .compare-sugg {
        background: rgba(16, 185, 129, 0.08);
        border-left: 3px solid #10B981;
    }
    </style>
    """, unsafe_allow_html=True)


def extract_uploaded_file(uploaded_file) -> tuple[str, str]:
    """Extract text from TXT, PDF, or DOCX without automatic OCR."""
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()
    if suffix == ".txt":
        try:
            return raw.decode("utf-8-sig"), uploaded_file.name
        except UnicodeDecodeError:
            return raw.decode("cp1252", errors="replace"), uploaded_file.name
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("PDF support is unavailable. Install packages in requirements.txt.")
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        extracted = "\n\n".join(pages).strip()
        if len(extracted.split()) < 10:
            raise ValueError(
                "This PDF produced little or no selectable text. It may be scanned. "
                "Run OCR or copy the text into a .txt file, check it carefully, and upload that file."
            )
        return extracted, uploaded_file.name
    if suffix == ".docx":
        if Document is None:
            raise RuntimeError("DOCX support is unavailable. Install packages in requirements.txt.")
        doc = Document(io.BytesIO(raw))
        chunks = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    chunks.append(" | ".join(cells))
        extracted = "\n".join(chunks).strip()
        if len(extracted.split()) < 5:
            raise ValueError("No meaningful text was found in this DOCX file.")
        return extracted, uploaded_file.name
    raise ValueError("Unsupported file type. Upload TXT, PDF or DOCX.")


def load_synthetic_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=["id", "title", "document_type", "text", "demo_scenario", "description"])


def load_model_metrics() -> dict:
    if MODEL_METRICS_PATH.exists():
        try:
            return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def build_review_csv(result: dict, doc_name: str) -> pd.DataFrame:
    rows = []
    for key, item in result["elements"].items():
        rows.append({
            "record_type": "structural_check",
            "document": doc_name,
            "item": item["label"],
            "status": "not applicable" if not item["applicable"] else ("cue detected" if item["present"] else "no cue detected"),
            "evidence": " | ".join(item["evidence"]),
            "score": "",
            "note": "Keyword/pattern screening only; presence does not prove clinical adequacy.",
        })
    for item in result["jargon"]:
        rows.append({
            "record_type": "jargon_flag", "document": doc_name, "item": item["term"],
            "status": "review suggested", "evidence": " | ".join(item["evidence"]),
            "score": item["count"], "note": f"Plain-language explanation: {item['plain_language']}",
        })
    for group in result["risk_language"]:
        for ev in group["evidence"]:
            rows.append({
                "record_type": "risk_language_flag", "document": doc_name, "item": group["category"],
                "status": "review for clarity", "evidence": ev, "score": "",
                "note": "Presence is not proof of defect; reviewer should confirm disclosure is clear and accurate.",
            })
    for issue in result["sentence_issues"]:
        rows.append({
            "record_type": "sentence_recommendation", "document": doc_name,
            "item": f"Sentence {issue['sentence_number']}", "status": issue["status"],
            "evidence": issue["original"], "score": issue["word_count"],
            "note": issue["suggestion"] or issue["note"],
        })
    nlp_data = result.get("nlp_analysis", {})
    for p in nlp_data.get("sentence_predictions", []):
        if p.get("is_flagged"):
            tokens_str = ", ".join(t["token"] for t in p.get("top_attributions", []))
            rows.append({
                "record_type": "nlp_classifier_flag", "document": doc_name,
                "item": f"Sentence {p['sentence_number']}",
                "status": f"Flagged ({p['predicted_issue_type']})",
                "evidence": p["sentence"],
                "score": f"{p['review_probability']:.1%}",
                "note": f"Model confidence: {p.get('confidence', 0):.1%}. Contributing tokens: {tokens_str}",
            })
    summary = summary_record(result, doc_name)
    for key, value in summary.items():
        rows.append({"record_type": "summary_metric", "document": doc_name, "item": key,
                    "status": "computed", "evidence": "", "score": value, "note": ""})
    return pd.DataFrame(rows)


# Inject custom styling
inject_custom_styles()

# Render modern Hero Banner
st.markdown("""
<div class="hero-card">
    <div class="hero-title">
        <span>🩺</span>
        <span>Informed Consent Comprehension & Risk Evaluation</span>
    </div>
    <div class="hero-subtitle">
        Clinical AI decision-support platform combining validated readability formulas, structural element auditing, and a supervised NLP classifier with local explainability (XAI).
    </div>
    <div class="hero-badges">
        <span class="hero-badge">Hybrid AI Architecture</span>
        <span class="hero-badge">TF-IDF + Logistic Regression</span>
        <span class="hero-badge">Local Feature Attribution (XAI)</span>
        <span class="hero-badge">Flesch-Kincaid & ARI</span>
        <span class="hero-badge">Version 2.0.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Analysis Parameters")
    target_grade = st.slider(
        "Target Reading Grade Level", min_value=5.0, max_value=12.0, value=8.0, step=0.5,
        help="Recommended general adult patient reading target is between Grade 6.0 and 8.0.",
    )
    procedure_type = st.selectbox(
        "Consent / Procedure Type",
        ["Research study", "Clinical treatment", "Surgical consent", "Off-label drug use"]
    )
    st.markdown("---")
    st.markdown("### 📋 Conditional Sections")
    reproductive_applicable = st.checkbox("Pregnancy / reproductive section is applicable", value=False)
    off_label_applicable = st.checkbox(
        "Off-label drug use explanation is required",
        value=(procedure_type == "Off-label drug use")
    )
    use_nlp = st.toggle("Supervised NLP Classifier", value=True, help="Enable machine-learned sentence triage & XAI.")
    show_preprocessing = st.checkbox("Show tokenizer details", value=False)

    st.markdown("---")
    st.markdown("### 🛡️ Ethical Safeguards")
    st.caption(
        "• SRI and ML scores do not predict individual comprehension.\n"
        "• Human IRB/clinical adjudication is always mandatory.\n"
        "• All processing is performed locally without external APIs."
    )

# Navigation Tabs
analysis_tab, studio_tab, batch_tab, examples_tab, methodology_tab = st.tabs([
    "📄 Document Evaluation",
    "🤖 NLP Classifier Studio",
    "📊 Batch CSV Audit",
    "🧪 Synthetic Test Library",
    "📚 Methodology & Architecture",
])

# -------------------------------------------------------------
# TAB 1: DOCUMENT EVALUATION
# -------------------------------------------------------------
with analysis_tab:
    st.markdown("### 1. Ingest Consent Document")
    
    upload_col, preset_col = st.columns([1, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "Upload Document (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"],
            help="Selectable text only. Scanned PDFs must be OCR-processed beforehand."
        )
    with preset_col:
        examples_df = load_synthetic_data()
        if not examples_df.empty:
            choice_map = {f"{row['id']} · {row['title']}": row["text"] for _, row in examples_df.iterrows()}
            selected_example = st.selectbox("Or choose a pre-loaded synthetic scenario:", ["(Select scenario)"] + list(choice_map.keys()))
            if selected_example != "(Select scenario)" and st.button("📥 Load Selected Scenario"):
                st.session_state["consent_text"] = choice_map[selected_example]
                st.session_state.pop("analysis_result", None)
                st.rerun()

    default_text = SAMPLE_PATH.read_text(encoding="utf-8") if SAMPLE_PATH.exists() else "Purpose: This study evaluates a new treatment. Risks: Side effects may occur. You may withdraw at any time."
    if "consent_text" not in st.session_state:
        st.session_state["consent_text"] = default_text

    pasted = st.text_area(
        "Consent Document Text (Edit or paste freely):", key="consent_text", height=240,
        help="This text is evaluated when you click Run Evaluation below."
    )
    if uploaded is not None:
        st.info(f"Selected file: **{uploaded.name}** (will be analyzed instead of the text box above).")

    run_col1, run_col2 = st.columns([2, 1])
    with run_col1:
        analyze_clicked = st.button("🚀 Run Comprehensive AI Evaluation", type="primary", use_container_width=True)
    with run_col2:
        if st.button("🔄 Reset to Default Sample", use_container_width=True):
            st.session_state["consent_text"] = default_text
            st.session_state.pop("analysis_result", None)
            st.rerun()

    if analyze_clicked:
        with st.spinner("Executing readability formulas, checklist audit, and NLP classifier..."):
            try:
                if uploaded is not None:
                    source_text, doc_name = extract_uploaded_file(uploaded)
                else:
                    source_text, doc_name = pasted, "pasted_text"
                result = analyze_consent(
                    source_text,
                    target_grade=target_grade,
                    procedure_type=procedure_type,
                    reproductive_applicable=reproductive_applicable,
                    off_label_applicable=off_label_applicable,
                    use_nlp_classifier=use_nlp,
                )
                st.session_state["analysis_result"] = result
                st.session_state["analysis_source_text"] = source_text
                st.session_state["analysis_doc_name"] = doc_name
                st.session_state["analysis_export_csv"] = build_review_csv(result, doc_name)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("analysis_result")
    if result:
        doc_name = st.session_state.get("analysis_doc_name", "pasted_text")
        nlp_data = result.get("nlp_analysis", {})

        st.markdown("---")
        st.markdown(f"### 2. Multi-Dimensional Evaluation Dashboard · `{doc_name}`")

        # Top KPI Metric Cards
        grade_diff = result['flesch_kincaid_grade'] - target_grade
        grade_sub = f"{'+' if grade_diff > 0 else ''}{grade_diff:.1f} vs Target ({target_grade:.1f})"
        nlp_flag_rate = nlp_data.get('flagged_share', 0.0) if nlp_data.get('classifier_active') else 0.0

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-icon">📘</div>
                <div class="kpi-label">Flesch–Kincaid</div>
                <div class="kpi-value" style="color: #38BDF8 !important;">{result['flesch_kincaid_grade']:.1f}</div>
                <div class="kpi-sub">{grade_sub}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🔬</div>
                <div class="kpi-label">Gunning Fog</div>
                <div class="kpi-value" style="color: #C084FC !important;">{result['gunning_fog_index']:.1f}</div>
                <div class="kpi-sub">ARI: {result.get('automated_readability_index', 0.0):.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📋</div>
                <div class="kpi-label">Completeness</div>
                <div class="kpi-value" style="color: #34D399 !important;">{result['structural_completeness_score']:.0f}%</div>
                <div class="kpi-sub">{result['present_element_count']}/{result['applicable_element_count']} elements detected</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🛡️</div>
                <div class="kpi-label">Semantic Risk Index</div>
                <div class="kpi-value" style="color: #F87171 !important;">{result['semantic_risk_index']}<span style="font-size:16px;color:#94a3b8">/100</span></div>
                <div class="kpi-sub">Priority: {result['semantic_risk_status']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🤖</div>
                <div class="kpi-label">NLP Model Flags</div>
                <div class="kpi-value" style="color: #FBBF24 !important;">{nlp_flag_rate:.0%}</div>
                <div class="kpi-sub">{nlp_data.get('flagged_count', 0)}/{nlp_data.get('total_sentences', 0)} sentences flagged</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Priority Triage Banner
        status = result["semantic_risk_status"]
        status_css = "triage-red" if status == "RED" else ("triage-yellow" if status == "YELLOW" else "triage-green")
        icon = "🚨" if status == "RED" else ("⚠️" if status == "YELLOW" else "✅")

        st.markdown(f"""
        <div class="triage-banner {status_css}">
            <div style="font-size:24px;">{icon}</div>
            <div>
                <b>REVIEW PRIORITY: {status}</b><br>
                {html.escape(result['semantic_risk_status_text'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabbed Results Breakdown
        res_tab1, res_tab2, res_tab3, res_tab4, res_tab5 = st.tabs([
            "🤖 Supervised NLP Classifier",
            "✍️ Sentence Issues & Rewrites",
            "📋 Checklist Audit",
            "🩺 Jargon & Risk Lexicon",
            "📊 Readability & SRI Breakdown",
        ])

        with res_tab1:
            st.markdown("#### 🤖 Supervised Sentence-Level Classifier Findings")
            if nlp_data.get("classifier_active"):
                m_info_col, tag_col = st.columns([2, 1])
                with m_info_col:
                    st.caption(f"**Model:** {nlp_data.get('model_type')} · **Test Accuracy:** {nlp_data.get('heldout_accuracy', 1.0):.1%} · **Macro-F1:** {nlp_data.get('heldout_macro_f1', 1.0):.3f}")
                with tag_col:
                    issue_counts = nlp_data.get("issue_counts", {})
                    if issue_counts:
                        pills = " ".join([f"<span class='token-chip token-red'>{k.replace('_', ' ').title()}: {v}</span>" for k, v in issue_counts.items()])
                        st.markdown(pills, unsafe_allow_html=True)

                preds = nlp_data.get("sentence_predictions", [])
                filter_choice = st.radio("Filter Sentences:", ["All Sentences", "Flagged by Model Only", "Clear Only"], horizontal=True)
                
                filtered = preds
                if filter_choice == "Flagged by Model Only":
                    filtered = [p for p in preds if p["is_flagged"]]
                elif filter_choice == "Clear Only":
                    filtered = [p for p in preds if not p["is_flagged"]]

                for p in filtered:
                    badge_class = "badge-review" if p["is_flagged"] else "badge-clear"
                    label_text = f"Needs Review ({p['review_probability']:.0%})" if p["is_flagged"] else f"Clear ({1.0 - p['review_probability']:.0%})"
                    issue_tag = p.get('predicted_issue_type', 'none').replace('_', ' ').title()

                    attributions = p.get("top_attributions", [])
                    token_pills = ""
                    for a in attributions:
                        c_class = "token-red" if a["attribution"] > 0 else "token-green"
                        token_pills += f"<span class='token-chip {c_class}'><b>{html.escape(a['token'])}</b> ({a['attribution']:+.2f})</span>"

                    st.markdown(f"""
                    <div class="sentence-card">
                        <div class="sentence-header">
                            <div><b>Sentence {p['sentence_number']}</b> · <span class="{badge_class}">{label_text}</span> · <span style="font-size:12px;color:#64748b;">Category: <b>{issue_tag}</b></span></div>
                            <div style="font-size:12px;color:#94a3b8">Confidence: {p.get('confidence', 0):.0%}</div>
                        </div>
                        <div style="font-size:14px;line-height:1.5;margin-bottom:8px;">{html.escape(p['sentence'])}</div>
                        <div>{'<b>Predictive Tokens:</b> ' + token_pills if token_pills else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("The supervised NLP classifier is currently disabled.")

        with res_tab2:
            st.markdown("#### ✍️ Flagged Sentences & Plain-Language Suggested Rewrites")
            if result["sentence_issues"]:
                for issue in result["sentence_issues"]:
                    with st.expander(f"Sentence {issue['sentence_number']} · {'; '.join(issue['issues'])}"):
                        st.markdown(f"""
                        <div class="compare-container">
                            <div class="compare-box compare-orig">
                                <b style="color:#64748B;">ORIGINAL SENTENCE:</b><br>
                                {html.escape(issue['original'])}
                            </div>
                            <div class="compare-box compare-sugg">
                                <b style="color:#10B981;">{'SUGGESTED REWRITE:' if issue['suggestion'] else 'REVIEWER ACTION:'}</b><br>
                                {html.escape(issue['suggestion'] if issue['suggestion'] else issue['note'])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if issue.get("changed_terms"):
                            st.caption(f"Replaced terms: {', '.join(issue['changed_terms'])}")
            else:
                st.success("No sentence-level rule issues were detected.")

        with res_tab3:
            st.markdown("#### 📋 Required Core & Conditional Elements Checklist")
            chk_rows = []
            for key, item in result["elements"].items():
                status_text = "⚪ Not Applicable" if not item["applicable"] else ("✅ Cue Detected" if item["present"] else "❌ Missing Cue — Inspect")
                chk_rows.append({
                    "Required Consent Element": item["label"],
                    "Status": status_text,
                    "Detected Text Evidence": " | ".join(item["evidence"]) if item["evidence"] else "(No matching text cues found)",
                    "Matches": item["match_count"],
                })
            st.dataframe(pd.DataFrame(chk_rows), hide_index=True, use_container_width=True)

        with res_tab4:
            j_col, r_col = st.columns(2)
            with j_col:
                st.markdown("#### 🩺 Medical Jargon Lexicon Matches")
                if result["jargon"]:
                    st.dataframe(pd.DataFrame([
                        {"Jargon Term": x["term"], "Count": x["count"], "Plain-Language Substitution": x["plain_language"]}
                        for x in result["jargon"]
                    ]), hide_index=True, use_container_width=True)
                else:
                    st.success("No lexicon medical jargon detected.")
            with r_col:
                st.markdown("#### ⚠️ Uncertainty & Exculpatory Phrases")
                if result["risk_language"]:
                    for item in result["risk_language"]:
                        with st.expander(f"{item['category']} ({item['count']} match(es))"):
                            for ev in item["evidence"]:
                                st.write(f"• {ev}")
                else:
                    st.success("No uncertainty or legalistic phrases detected.")

        with res_tab5:
            rb_col, sri_col = st.columns(2)
            with rb_col:
                st.markdown("#### Readability Metrics")
                r_df = pd.DataFrame([
                    {"Metric": "Flesch–Kincaid Grade Level", "Score": f"{result['flesch_kincaid_grade']:.2f}"},
                    {"Metric": "Gunning Fog Index", "Score": f"{result['gunning_fog_index']:.2f}"},
                    {"Metric": "Automated Readability Index (ARI)", "Score": f"{result.get('automated_readability_index', 0.0):.2f}"},
                    {"Metric": "Flesch Reading Ease", "Score": f"{result['flesch_reading_ease']:.2f}"},
                    {"Metric": "Long sentences (>25 words)", "Score": str(result['long_sentence_count'])},
                    {"Metric": "Complex words (3+ syllables)", "Score": str(result['complex_word_count'])},
                ])
                st.dataframe(r_df, hide_index=True, use_container_width=True)
            with sri_col:
                st.markdown("#### Semantic Risk Index Component Penalties")
                s_df = pd.DataFrame([{"Component": k, "Points": v} for k, v in result["sri_components"].items()])
                st.dataframe(s_df, hide_index=True, use_container_width=True)

        # Download Buttons
        st.markdown("---")
        st.markdown("#### 💾 Audit & Report Exports")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        export_df = st.session_state["analysis_export_csv"]
        with exp_col1:
            st.download_button(
                "📥 Download Detailed Audit CSV",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{doc_name}_detailed_audit.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_col2:
            st.download_button(
                "📥 Download Complete Results JSON",
                data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{doc_name}_analysis.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp_col3:
            summary_df = pd.DataFrame([summary_record(result, doc_name)])
            st.download_button(
                "📥 Download One-Row Summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{doc_name}_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

# -------------------------------------------------------------
# TAB 2: NLP CLASSIFIER STUDIO
# -------------------------------------------------------------
with studio_tab:
    st.markdown("### 🤖 NLP Classifier Studio & Explainability (XAI)")
    st.caption("Real-time sentence triage sandbox with local feature attribution ($w_i \\cdot x_i$) and global model diagnostics.")

    clf = SemanticConsentClassifier.get_default_instance()
    metrics = load_model_metrics()

    if not clf.is_available():
        st.error("Model artifact not found. Run `python train_semantic_classifier.py` to compile the model.")
    else:
        st.markdown("#### 🔬 Live Interactive Sentence Classifier Sandbox")
        presets = {
            "Plain language voluntariness": "Taking part in this study is completely voluntary and you may stop at any time without penalty.",
            "Plain language clinical procedure": "You will take two pills by mouth every morning with a glass of water.",
            "Medical jargon (cardiovascular)": "The patient will receive intravenous infusion of the compound with monitoring for hemodynamic instability and myocardial ischemia.",
            "Medical jargon (pharmacokinetics)": "Serial pharmacodynamic and pharmacokinetic biomarker evaluations will be conducted according to protocol.",
            "Legalistic liability waiver": "The undersigned participant hereby agrees to waive all claims and hold harmless the clinical trial sponsor.",
            "Legalistic indemnification": "You agree to indemnify and hold harmless the institution notwithstanding any adverse outcomes.",
            "Unclear risk & uncertainty": "The experimental treatment may cause unpredictable adverse complications of uncertain incidence that are not fully understood.",
            "Convoluted syntax / long sentence": "Prior to initiation of therapeutic administration, prospective subjects must undergo comprehensive baseline screening consisting of biochemical evaluations and radiographic assessments.",
        }

        preset_choice = st.selectbox("Select a benchmark sample sentence:", ["(Custom input)"] + list(presets.keys()))
        default_val = presets[preset_choice] if preset_choice != "(Custom input)" else "The undersigned participant hereby agrees to waive all claims and hold harmless the sponsor."
        sandbox_text = st.text_area("Input sentence to classify:", value=default_val, height=80)

        if st.button("⚡ Classify Sentence with NLP Model", type="primary"):
            if sandbox_text.strip():
                pred = clf.predict_sentence(sandbox_text.strip())
                attributions = clf.explain_sentence(sandbox_text.strip(), top_n=6)

                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    if pred["is_flagged"]:
                        st.markdown("""<div class="triage-banner triage-red" style="margin:0;">🚨 <b>LABEL: NEEDS REVIEW</b></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="triage-banner triage-green" style="margin:0;">✅ <b>LABEL: CLEAR</b></div>""", unsafe_allow_html=True)
                with sc2:
                    st.metric("Review Probability", f"{pred['review_probability']:.1%}")
                with sc3:
                    st.metric("Predicted Issue Subcategory", pred["predicted_issue_type"].replace("_", " ").title())

                st.progress(pred["review_probability"])

                st.markdown("##### 🔍 Local Token Attribution Badges")
                st.caption("Positive attribution scores (red) push the model towards flagging; negative scores (green) push towards clear.")
                if attributions:
                    pills = []
                    for a in attributions:
                        c_class = "token-red" if a["attribution"] > 0 else "token-green"
                        pills.append(f"<span class='token-chip {c_class}'><b>{html.escape(a['token'])}</b>: {a['attribution']:+.3f}</span>")
                    st.markdown(" ".join(pills), unsafe_allow_html=True)
                else:
                    st.info("No specific vocabulary n-grams had dominant attribution weights for this sentence.")

        st.markdown("---")
        st.markdown("#### 📈 Global Model Performance & Architecture")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Model Architecture", "TF-IDF + Logistic Regression")
        g2.metric("Held-Out Accuracy", f"{metrics.get('accuracy', 1.0):.1%}")
        g3.metric("Held-Out Macro-F1", f"{metrics.get('macro_f1', 1.0):.3f}")
        g4.metric("Validation Protocol", "GroupShuffleSplit (Document-Level)")

        f_left, f_right = st.columns(2)
        with f_left:
            st.markdown("##### Top Predictive N-Grams for 'Needs Review'")
            top_rev = metrics.get("top_features", {}).get("needs_review", [])
            if top_rev:
                st.dataframe(pd.DataFrame(top_rev), hide_index=True, use_container_width=True)
        with f_right:
            st.markdown("##### Top Predictive N-Grams for 'Clear'")
            top_clr = metrics.get("top_features", {}).get("clear", [])
            if top_clr:
                st.dataframe(pd.DataFrame(top_clr), hide_index=True, use_container_width=True)

        if "confusion_matrix" in metrics:
            st.markdown("##### Confusion Matrix (Held-Out Test Set)")
            cm = metrics["confusion_matrix"]
            classes = metrics.get("classes", ["clear", "needs_review"])
            cm_df = pd.DataFrame(cm, index=[f"Actual: {c}" for c in classes], columns=[f"Predicted: {c}" for c in classes])
            st.dataframe(cm_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: BATCH CSV AUDIT
# -------------------------------------------------------------
with batch_tab:
    st.markdown("### 📊 Batch Evaluation of Multiple Consent Forms")
    st.write("Upload a CSV file containing multiple consent documents for automated batch screening.")
    batch_file = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch_uploader")
    if batch_file is not None:
        try:
            b_df = pd.read_csv(batch_file)
            preferred = [c for c in b_df.columns if c.strip().lower() in {"text", "consent_text", "document", "content", "consent"}]
            default_idx = b_df.columns.get_loc(preferred[0]) if preferred else 0
            text_col = st.selectbox("Select Consent Text Column:", list(b_df.columns), index=default_idx)
            name_options = ["(row number)"] + [c for c in b_df.columns if c != text_col]
            name_col = st.selectbox("Optional Identifier Column:", name_options)

            if st.button("🚀 Process Batch Documents", type="primary"):
                with st.spinner("Processing documents in batch..."):
                    batch_res = []
                    for idx, row in b_df.iterrows():
                        doc_text = "" if pd.isna(row[text_col]) else str(row[text_col])
                        doc_id = str(idx + 1) if name_col == "(row number)" else str(row[name_col])
                        try:
                            analysis = analyze_consent(
                                doc_text,
                                target_grade=target_grade,
                                procedure_type=procedure_type,
                                reproductive_applicable=reproductive_applicable,
                                off_label_applicable=off_label_applicable,
                                use_nlp_classifier=use_nlp,
                            )
                            batch_res.append(summary_record(analysis, doc_id))
                        except Exception as exc:
                            batch_res.append({"document_name": doc_id, "error": str(exc)})
                    st.session_state["batch_results"] = batch_res

            b_results = st.session_state.get("batch_results")
            if b_results:
                res_df = pd.DataFrame(b_results)
                st.success(f"Successfully processed {len(b_results)} documents.")
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download Batch Results CSV",
                    data=res_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="batch_consent_audit.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")

# -------------------------------------------------------------
# TAB 4: SYNTHETIC TEST LIBRARY
# -------------------------------------------------------------
with examples_tab:
    st.markdown("### 🧪 Synthetic Consent Benchmark Scenarios")
    st.caption("Invented demonstration texts illustrating varying complexity profiles for testing and IRB training.")
    example_df = load_synthetic_data()
    st.dataframe(example_df[["id", "title", "document_type", "demo_scenario", "description"]], use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 5: METHODOLOGY & ARCHITECTURE
# -------------------------------------------------------------
with methodology_tab:
    st.markdown("### 📚 Hybrid Evaluation Methodology")
    st.markdown(r"""
#### 1. Hybrid Multi-Layer Architecture
- **Deterministic Readability Layer:** Computes Flesch–Kincaid Grade Level, Gunning Fog, and Automated Readability Index (ARI).
- **Structural Integrity Checklist:** Evaluates presence of 7 core elements (Purpose, Procedure, Risks, Benefits, Alternatives, Monitoring, Autonomy) and 2 conditional disclosures.
- **Lexical Pattern Engine:** Identifies 50+ medical jargon terms and legalistic rights-waiving patterns.
- **Supervised Machine Learning Layer:** TF-IDF n-gram vectorizer paired with a class-balanced Logistic Regression classifier trained on multi-domain clinical consent sentences.
- **Explainable AI (XAI):** Linear attribution ($w_i \cdot x_i$) pinpoints exact tokens influencing review scores.

#### 2. Semantic Risk Index (SRI) Formula
$$SRI = \min(100, P_{\text{grade}} + P_{\text{length}} + P_{\text{jargon}} + P_{\text{uncertainty}} + P_{\text{missing\_elements}} + P_{\text{critical\_autonomy}})$$

#### 3. Ethics & Responsible AI Disclosure
- This tool is designed strictly for reviewer decision-support and does not establish legal sufficiency or guarantee patient comprehension.
- Local processing ensures patient confidentiality is maintained.
""")

st.markdown("---")
st.caption(f"Informed Consent Comprehension & Risk Evaluation · Version {__import__('consent_analyzer').APP_VERSION} · Confidential & Local")
