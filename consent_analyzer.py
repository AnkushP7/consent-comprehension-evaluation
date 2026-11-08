"""Transparent, deterministic NLP baseline and supervised classifier for consent-document review.

This module provides screening indicators combining rule-based checks with
supervised machine learning to support human reviewers. It does not provide clinical
judgments or replace institutional ethics review.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

try:
    import textstat  # type: ignore
except ImportError:  # Tests/core can still run before dependency installation.
    textstat = None

from nlp_classifier import SemanticConsentClassifier

APP_VERSION = "2.0.0"

# Required for most consent documents. Presence is determined by transparent cues;
# it does not mean the element is legally sufficient or clinically adequate.
CORE_ELEMENTS: dict[str, dict[str, Any]] = {
    "purpose": {
        "label": "Purpose and context",
        "patterns": [
            r"\bpurpose\b", r"\baim of (?:the )?study\b", r"\bobjective of\b",
            r"\bthis study (?:aims|is designed|will)\b", r"\bwe are conducting this study\b",
            r"\bwe want to learn\b", r"\bdesigned to help\b", r"\bprimary objective\b",
        ],
    },
    "procedure": {
        "label": "Procedure / study activities",
        "patterns": [
            r"\bprocedure(?:s)?\s*:", r"\bsurgery\b", r"\boperation\b",
            r"\byou will (?:receive|take|undergo|be asked to|have)\b", r"\bstudy involves\b",
            r"\bthe study will (?:involve|require|include)\b", r"\byour blood will be (?:drawn|collected)\b",
            r"\bblood draw(?:s)?\b", r"\b(?:a|one|two|three) blood samples? (?:will be )?(?:collected|taken)\b",
            r"\bwe will collect (?:(?:a|one|two|three)\s+)?blood sample\b", r"\bscan(?:s)? will be performed\b",
            r"\bquestionnaires? will be (?:given|completed)\b", r"\binjection in your\b",
            r"\bwill make (?:three|four|\d+) tiny cuts\b",
        ],
    },
    "risks": {
        "label": "Risks and side effects",
        "patterns": [
            r"\brisk(?:s)?\s+(?:include|may include|are|of|associated with|to you)\b",
            r"\b(?:possible|potential|known|expected) risks?\b", r"\bside effects?\b",
            r"\badverse (?:event|effect)s?\b", r"\bcomplication(?:s)?\b", r"\bbleeding\b",
            r"\binfection\b", r"\bmay cause\b", r"\bcould cause\b", r"\bdiscomfort\b",
            r"\bpain at the (?:site|area)\b", r"\bharm an unborn baby\b",
        ],
    },
    "benefits": {
        "label": "Expected benefits",
        "patterns": [
            r"\bbenefit(?:s|ed)?\b", r"\bmay help\b", r"\bmay improve\b",
            r"\bexpected to improve\b", r"\bno direct benefit\b", r"\bcannot promise this outcome\b",
        ],
    },
    "alternatives": {
        "label": "Alternatives, including no treatment",
        "patterns": [
            r"\balternatives?\b", r"\bother (?:treatment|options?|pain medicines)\b",
            r"\bstandard (?:care|treatment)\b", r"\bno treatment\b", r"\bchoose not to participate\b",
            r"\bchoose other treatments\b",
        ],
    },
    "monitoring": {
        "label": "Monitoring and follow-up",
        "patterns": [
            r"\bmonitor(?:ed|ing)?\b", r"\bfollow[ -]?up\b", r"\bregular (?:visits|checks|assessments)\b",
            r"\bcontact the (?:study )?team\b", r"\bwill be checked\b", r"\bvital signs\b",
            r"\bcheck your blood sugar\b", r"\btest your memory\b",
        ],
    },
    "autonomy": {
        "label": "Voluntary participation / refusal / withdrawal",
        "patterns": [
            r"\bvoluntary\b", r"\bright to refuse\b", r"\bmay refuse\b", r"\bwithdraw\b",
            r"\bwithout (?:penalty|affecting)\b", r"\bmay stop at any time\b", r"\bask questions\b",
            r"\byou can choose\b", r"\bfree to stop\b",
        ],
    },
}

CONDITIONAL_ELEMENTS: dict[str, dict[str, Any]] = {
    "reproductive": {
        "label": "Reproductive / pregnancy considerations",
        "patterns": [
            r"\bpregnan(?:t|cy)\b", r"\bbreastfeed(?:ing)?\b", r"\bcontraception\b",
            r"\bfertil(?:ity|e)\b", r"\bunborn child\b", r"\bunborn baby\b", r"\breproductive\b",
            r"\bbirth control\b",
        ],
    },
    "off_label": {
        "label": "Off-label-use explanation",
        "patterns": [
            r"\boff[ -]label\b", r"\bnot approved\b", r"\bapproved for another use\b",
            r"\bunapproved use\b",
        ],
    },
}

JARGON: dict[str, str] = {
    "contraindication": "a reason the treatment should not be used",
    "contraindicated": "strongly advised against because it could cause harm",
    "pharmacokinetics": "how the body absorbs, processes, and removes a medicine",
    "pharmacodynamic": "how the medicine affects the body",
    "pharmacodynamics": "how the medicine affects the body",
    "hemodynamic": "related to blood flow and blood pressure",
    "myocardial": "related to the heart muscle",
    "intravenous": "given through a vein",
    "subcutaneous": "given under the skin",
    "intramuscular": "given into a muscle",
    "adverse event": "an unwanted medical event or side effect",
    "adverse events": "unwanted medical events or side effects",
    "comorbidity": "another health condition present at the same time",
    "etiology": "the cause of a disease or condition",
    "prophylaxis": "treatment used to prevent a disease or condition",
    "prophylactic": "preventive treatment",
    "ischemia": "reduced blood flow to a body part",
    "immunogenicity": "the ability to trigger an immune response",
    "placebo": "an inactive substance (sugar pill) with no active medicine",
    "randomized": "assigned by chance",
    "randomisation": "assignment to a treatment group by chance",
    "double-blind": "neither you nor key study staff know which treatment you receive",
    "open-label": "an open study where both you and the doctors know which medicine you receive",
    "multicentre": "conducted across multiple hospitals",
    "incidence": "how often something happens",
    "protocol": "the planned rules and steps of a study",
    "de-identified": "with identifying information removed",
    "etiological": "related to the cause of a condition",
    "malignancy": "cancer or another harmful growth",
    "hypertension": "high blood pressure",
    "hypotension": "low blood pressure",
    "orthostatic hypotension": "low blood pressure when standing up, which can cause dizziness",
    "edema": "swelling caused by fluid",
    "lesion": "an area of damaged or abnormal tissue",
    "invasive": "involving entry into the body",
    "prognosis": "the expected course or outcome of a condition",
    "compliance": "following the treatment plan or instructions",
    "discontinue": "stop",
    "administered": "given",
    "eligibility criteria": "the requirements to take part",
    "sequelae": "long-term conditions or after-effects of a disease or treatment",
    "toxicities": "harmful or poisonous side effects",
    "toxicity": "harmful or poisonous side effects",
    "titration": "adjusting the medicine dose step by step",
    "down-titration": "gradually decreasing the medicine dose",
    "down-titrate": "gradually decrease the dose",
    "efficacy": "how well a treatment works",
    "adjunctive therapy": "additional add-on treatment taken alongside your current medicine",
    "adjunctive": "additional add-on",
    "pulmonary arterial hypertension": "high blood pressure in the arteries of the lungs (PAH)",
    "washout period": "a break from treatment to clear medicines from your body",
    "biopsy": "taking a small tissue sample to test in a lab",
    "lumbar puncture": "a spinal tap to collect fluid around your spine",
    "refractory": "not responding to standard treatment",
    "remission": "a decrease or disappearance of disease symptoms",
    "anaphylaxis": "a severe, potentially life-threatening allergic reaction",
    "concomitant": "taken or happening at the same time",
    "exculpatory": "relieving someone from blame or legal liability",
    "indemnify": "agreeing to pay someone else's legal losses or costs",
    "irrevocable": "permanent and cannot be undone or changed",
    "thrombosis": "formation of a blood clot inside a blood vessel",
    "embolization": "blocking of a blood vessel by a traveling clot or object",
    "spirometry": "a breathing test that measures how much air you blow out",
    "catheterisation": "inserting a thin, flexible tube into a blood vessel",
    "catheterization": "inserting a thin, flexible tube into a blood vessel",
    "tamponade": "fluid build-up around the heart that impairs normal pumping",
    "dyskinesia": "uncontrollable, involuntary body movements",
    "syncope": "fainting or temporary loss of consciousness",
    "hyperhidrosis": "excessive sweating",
    "pruritus": "severe skin itching",
    "dermatitis": "skin irritation or inflammation",
    "biomarker": "a natural substance in the blood used to measure health",
    "biomarkers": "natural substances in the blood used to measure health",
    "echocardiography": "an ultrasound scan of your heart",
}

# Signals for human review, not assertions that disclosure is wrong.
RISK_PATTERNS: dict[str, list[str]] = {
    "Uncertainty / unknown outcomes": [
        r"\bnot fully understood\b", r"\bunknown risks?\b", r"\buncertain (?:incidence|outcome|effect)\b",
        r"\bwe do not know (?:whether|if|how)\b", r"\bmay not benefit\b", r"\bno guarantee\b",
        r"\bresults? (?:are|is) uncertain\b", r"\bmay or may not\b", r"\bremain entirely unknown\b",
        r"\bunpredictable risks?\b", r"\bcannot be anticipated\b",
    ],
    "Legalistic / rights-related wording": [
        r"\bwaive(?:s|d|r)?\b", r"\bindemnif(?:y|ies|ication)\b", r"\bhold harmless\b",
        r"\bwithout prejudice to\b", r"\bhereby acknowledge\b", r"\bnotwithstanding\b",
        r"\birrevocably release\b", r"\barbitrate all claims\b", r"\bdisclaims all warranties\b",
        r"\bentire integrated agreement\b",
    ],
    "Strong risk disclosure (review for clarity)": [
        r"\blife[ -]threatening\b", r"\bpermanent disability\b", r"\bmay result in death\b",
        r"\bserious adverse events?\b", r"\bsevere complications?\b", r"\bcatastrophic complications?\b",
    ],
}

# More specific patterns first; substitutions are conservative and do not resolve
# clinical ambiguity. Unmatched issues are explicitly sent to human review.
REWRITES: list[tuple[str, str]] = [
    # Exculpatory warnings
    (r"\bhereby agrees? to waive (?:any and )?all claims\b", "[IRB Notice: Exculpatory waiver language should be removed]"),
    (r"\bagrees? to indemnify and hold harmless\b", "[IRB Notice: Indemnification clause should be removed for patient participants]"),
    (r"\birrevocably releases? the hospital\b", "[IRB Notice: Liability release should be removed]"),
    # Phrase-level clinical substitutions
    (r"\bpulmonary arterial hypertension\s*\(PAH\)\b", "high blood pressure in the arteries of your lungs (PAH)"),
    (r"\bpulmonary arterial hypertension\b", "high blood pressure in the arteries of your lungs"),
    (r"\borthostatic hypotension\b", "low blood pressure when standing up, which can cause dizziness"),
    (r"\bessential hypertension\b", "high blood pressure"),
    (r"\bhypertension\b", "high blood pressure"),
    (r"\bhypotension\b", "low blood pressure"),
    (r"\bsafety and efficacy\b", "safety and how well the treatment works"),
    (r"\btherapeutic efficacy\b", "how well the treatment works"),
    (r"\befficacy of\b", "how well the treatment works for"),
    (r"\befficacy\b", "how well the treatment works"),
    (r"\badjunctive therapy\b", "additional add-on treatment taken alongside your current medicine"),
    (r"\badjunctive\b", "additional add-on"),
    (r"\ban open-label, multicentre study\b", "a study where you and the doctors know which medicine is given, conducted across multiple hospitals"),
    (r"\bopen-label\b", "open study where treatments are known"),
    (r"\bmulticentre\b", "conducted across multiple hospitals"),
    (r"\bright[- ]heart catheterisation\b|\bright[- ]heart catheterization\b", "a procedure guiding a thin tube into the right side of the heart to measure blood pressures"),
    (r"\bcatheterisation\b|\bcatheterization\b", "inserting a thin, flexible tube into a blood vessel"),
    (r"\bcardiac tamponade\b", "fluid build-up around the heart that impairs normal pumping"),
    (r"\bair embolism\b", "an air bubble entering a blood vessel"),
    (r"\bpulmonary artery rupture\b", "damage or tearing in the main blood vessel leading to your lungs"),
    (r"\bdyskinesia\b", "uncontrollable, involuntary body movements"),
    (r"\bsyncope\b", "fainting or temporary loss of consciousness"),
    (r"\bhyperhidrosis\b", "excessive sweating"),
    (r"\bpruritus\b", "severe skin itching"),
    (r"\bdown-titrat(?:e|ion|ed|ing)\b", "gradually reducing the medicine dose"),
    (r"\bconcomitant medications?\b", "other medicines you take at the same time"),
    (r"\bconcomitant therapy\b", "other medicines you take at the same time"),
    (r"\bwashout period\b", "break from treatment to clear medicines from your body"),
    (r"\bdose titration\b", "gradually adjusting your medicine dose"),
    (r"\btherapeutic efficacy\b", "how well the treatment works"),
    (r"\bby subcutaneous injection\b", "by injection under the skin"),
    (r"\bsubcutaneous injection\b", "injection under the skin"),
    (r"\bthe pharmacokinetics of (?:the )?medicine\b", "how the body absorbs and processes the medicine"),
    (r"\bpharmacokinetics of (?:the )?medicine\b", "how the body absorbs and processes the medicine"),
    (r"\bpharmacokinetics of (?:the )?study drug\b", "how the body absorbs and processes the study medicine"),
    (r"\bunique protocol id\s*:?\b", "study identification number"),
    (r"\bprotocol number\b", "study identification number"),
    (r"\bstudy protocol\b", "study plan"),
    (r"\bprotocol\b", "study plan"),
    (r"\bsubcutaneous\b", "under the skin"),
    (r"\bintramuscular\b", "into a muscle"),
    (r"\bpharmacodynamic(?:s)?\b", "how the medicine affects the body"),
    (r"\bpharmacokinetics\b", "how the body absorbs and processes it"),
    (r"\bpharmacokinetic\b", "related to how the body absorbs and processes a medicine"),
    (r"\bmedroxyprogesterone acetate\b", "a hormone medicine called medroxyprogesterone acetate"),
    (r"\bovulation\b", "the release of an egg from an ovary"),
    (r"\bcoagulation\b", "blood clotting"),
    (r"\belectrocardiogram\b|\bECG\b", "a test that records the heart's electrical activity"),
    (r"\bhematologic(?:al)?\b", "related to blood"),
    (r"\bsolid tumors?\b", "cancerous growths"),
    (r"\bmetastatic\b", "cancer that has spread"),
    (r"\bplacebo\b", "an inactive substance (sugar pill) used for comparison"),
    (r"\brandomi[sz]ed\b", "assigned by chance"),
    (r"\bconcomitant(?:ly)?\b", "at the same time"),
    (r"\bdiscontinue\b", "stop"),
    (r"\badminister(?:ed|ing)?\b", "give"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
    (r"\bin the event that\b", "if"),
    (r"\bthere is no guarantee that you will benefit\b", "the treatment may not help you"),
    (r"\bother risks that are not fully understood\b", "other risks that are not yet fully known"),
    (r"\bby intravenous infusion\b", "through a vein"),
    (r"\bhemodynamic instability\b", "problems with blood pressure or blood flow"),
    (r"\bmyocardial ischemia\b", "reduced blood flow to the heart muscle"),
    (r"\bpossible adverse events\b", "possible unwanted side effects"),
    (r"\bserious adverse events?\b", "serious unwanted medical effects"),
    (r"\bthe investigational product\b", "the study medicine"),
    (r"\bnot fully understood\b", "not yet fully known"),
    (r"\bimmunogenicity\b", "the chance that the treatment triggers an immune response"),
    (r"\bcontraindication\b", "a reason the treatment should not be used"),
    (r"\bcontraindicated\b", "strongly advised against"),
    (r"\bprophylaxis\b", "treatment used to prevent a condition"),
    (r"\bcomorbidity\b", "another health condition"),
    (r"\bmyocardial\b", "related to the heart muscle"),
    (r"\bhemodynamic\b", "related to blood flow and blood pressure"),
    (r"\bintravenous\b", "given through a vein"),
    (r"\badverse events?\b", "unwanted medical effects"),
    (r"\bincidence\b", "how often something happens"),
    (r"\bischemia\b", "reduced blood flow"),
    (r"\butilize\b", "use"),
    (r"\bcommence\b", "start"),
    (r"\bterminate\b", "stop"),
    (r"\bapproximately\b", "about"),
    (r"\bwith regard to\b", "about"),
    (r"\bin order to\b", "to"),
    (r"\bdemonstrate\b", "show"),
    (r"\bindividuals\b", "people"),
    (r"\bparticipants\b", "people taking part in the study"),
    (r"\bundergo\b", "have"),
    (r"\bobtain\b", "get"),
]

SECTION_HEADER_RE = re.compile(
    r"(?im)^\s*(purpose(?: and context)?|study purpose|procedure(?:s)?|risks?(?: and side effects?)?|benefits?|alternatives?|monitoring|follow[ -]?up|voluntary participation|patient autonomy|off[ -]?label use|pregnancy|reproductive considerations)\s*:\s*"
)
WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b", re.UNICODE)


def clean_text(text: str) -> str:
    """Normalize line endings and whitespace while preserving paragraph boundaries."""
    text = (text or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def unwrap_soft_lines(text: str) -> str:
    """Join soft line breaks from PDF or text wrapping, preserving paragraph boundaries and list items."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    unwrapped = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if unwrapped and unwrapped[-1] != "":
                unwrapped.append("")
            continue
        # Filter obvious recurring PDF header/footer lines
        if re.match(r"(?im)^(Page\s+\d+\s+of\s+\d+|Patient Information Sheet|Final English Master|Protocol No\.:.*EudraCT|Bial\s*-\s*Portela\s*&|[_\s-]{12,})$", stripped):
            continue
        if not unwrapped or unwrapped[-1] == "":
            unwrapped.append(stripped)
        else:
            prev = unwrapped[-1]
            if prev.endswith("-") and not prev.endswith(" -"):
                unwrapped[-1] = prev[:-1] + stripped
            elif not re.search(r"[.!?:]\s*$", prev) and not re.match(r"^([•\-\*]|\d+[\.\)])\s+", stripped):
                unwrapped[-1] = prev + " " + stripped
            else:
                unwrapped.append(stripped)
    return "\n".join(unwrapped)


def split_sentences(text: str) -> list[str]:
    """Split simple English prose while treating section headers and bullet lists as boundaries."""
    unwrapped = unwrap_soft_lines(text)
    normalized = clean_text(unwrapped)
    normalized = SECTION_HEADER_RE.sub(lambda m: f"\n{m.group(1)}: ", normalized)
    # Common abbreviations should not trigger sentence boundaries.
    normalized = re.sub(r"\b(e\.g|i\.e|Dr|Mr|Mrs|Ms|vs|approx|etc|viz)\.", lambda m: m.group(1) + "<DOT>", normalized, flags=re.I)
    parts = re.split(r"(?<=[.!?])(?:[\"'’)]*)\s+|\n\s*[-•*]\s+|\n{2,}", normalized)
    result = []
    for part in parts:
        part = part.replace("<DOT>", ".").strip(" \t\n•-*")
        if part and len(words(part)) >= 2:
            result.append(part)
        elif part and any(re.search(p, part, re.I) for p in [r"\b(purpose|procedure|risks?|benefits?|alternatives?|monitoring|voluntary)\b"]):
            result.append(part)
    return result


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ye")):
        count -= 1
    if w.endswith("es") and count > 1 and not w.endswith(("ses", "zes", "ches", "shes")):
        count -= 1
    return max(1, count)


def _fallback_readability(text: str, sentences: list[str], tokens: list[str]) -> dict[str, float]:
    alpha_words = [t for t in tokens if re.search(r"[A-Za-z]", t)]
    n_words = max(1, len(alpha_words))
    n_sentences = max(1, len(sentences))
    n_chars = sum(len(w) for w in alpha_words)
    syllables = sum(_syllables(t) for t in alpha_words)
    complex_words = sum(_syllables(t) >= 3 for t in alpha_words)

    fk = 0.39 * (n_words / n_sentences) + 11.8 * (syllables / n_words) - 15.59
    fre = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (syllables / n_words)
    fog = 0.4 * ((n_words / n_sentences) + 100 * complex_words / n_words)
    ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sentences) - 21.43

    return {
        "flesch_kincaid_grade": max(0.0, fk),
        "gunning_fog_index": max(0.0, fog),
        "flesch_reading_ease": max(-50.0, min(120.0, fre)),
        "automated_readability_index": max(0.0, ari),
    }


def calculate_readability(text: str) -> dict[str, Any]:
    sentences = split_sentences(text)
    tokens = words(text)
    alpha_words = [t for t in tokens if re.search(r"[A-Za-z]", t)]
    n_words = len(tokens)
    n_sentences = len(sentences)
    base: dict[str, Any] = {
        "word_count": n_words,
        "sentence_count": n_sentences,
        "average_sentence_length": round(n_words / n_sentences, 2) if n_sentences else 0.0,
        "long_sentence_count": sum(len(words(s)) > 25 for s in sentences),
        "long_sentence_share": round(sum(len(words(s)) > 25 for s in sentences) / n_sentences, 3) if n_sentences else 0.0,
    }
    if n_words < 20 or n_sentences < 2:
        base["readability_warning"] = "Very short text: readability formulas are unstable on short inputs. Treat values as indicative only."
    else:
        base["readability_warning"] = None
    try:
        if textstat is None:
            raise RuntimeError("textstat is not installed")
        base.update({
            "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
            "gunning_fog_index": float(textstat.gunning_fog(text)),
            "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
            "automated_readability_index": float(textstat.automated_readability_index(text)),
        })
        base["readability_method"] = "textstat"
    except Exception:
        base.update(_fallback_readability(text, sentences, tokens))
        base["readability_method"] = "transparent fallback approximation"

    for key in ("flesch_kincaid_grade", "gunning_fog_index", "automated_readability_index"):
        value = base.get(key, 0.0)
        if not isinstance(value, (float, int)) or value != value:
            base[key] = 0.0
        base[key] = round(max(0.0, float(base[key])), 2)

    base["flesch_reading_ease"] = round(float(base["flesch_reading_ease"]), 2)
    base["complex_word_count"] = sum(_syllables(t) >= 3 for t in alpha_words)
    return base


def _is_template_metadata(sentence: str) -> bool:
    """Reject obvious form-template labels/instructions as evidence of disclosure."""
    s = sentence.strip().lower()
    if not s:
        return True
    if re.search(r"\b(suggested use|sample consent|template|example only|insert (?:study|text|details)|[\[<].{0,80}[\]>])\b", s):
        return True
    if re.fullmatch(r"\s*(purpose|procedure|risks?|benefits?|alternatives?|monitoring|follow[- ]?up|confidentiality|withdrawal)\s*:?\s*", s):
        return True
    return False


def detect_element(text: str, patterns: list[str]) -> dict[str, Any]:
    matches = []
    for sentence in split_sentences(text):
        if _is_template_metadata(sentence):
            continue
        if any(re.search(p, sentence, re.I) for p in patterns):
            matches.append(sentence)
    return {"present": bool(matches), "evidence": matches[:3], "match_count": len(matches)}


def detect_jargon(text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for term, explanation in JARGON.items():
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        matches = list(re.finditer(pattern, text, re.I))
        if matches:
            snippets = []
            for match in matches[:3]:
                start = max(0, text.rfind(".", 0, match.start()) + 1)
                end_candidates = [x for x in [text.find(".", match.end()), text.find("!", match.end()), text.find("?", match.end()), text.find("\n", match.end())] if x != -1]
                end = min(end_candidates) + 1 if end_candidates else min(len(text), match.end() + 100)
                snippets.append(text[start:end].strip())
            found.append({"term": term, "count": len(matches), "plain_language": explanation, "evidence": snippets})
    return found


def detect_risk_language(text: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for category, patterns in RISK_PATTERNS.items():
        matches = []
        for sentence in split_sentences(text):
            if any(re.search(p, sentence, re.I) for p in patterns):
                matches.append(sentence)
        if matches:
            result.append({"category": category, "count": len(matches), "evidence": matches[:5]})
    return result


def simplify_sentence(sentence: str) -> dict[str, Any]:
    """Return conservative rule-based rewrite; never pretend unchanged text was revised."""
    out = sentence.strip()
    changed_terms = []
    # 1. High-priority phrase substitutions
    for pattern, replacement in REWRITES:
        revised, count = re.subn(pattern, replacement, out, flags=re.I)
        if count:
            out = revised
            changed_terms.append(re.sub(r"\\[bB]|\\b", "", pattern).strip("^$"))

    # 2. Term-level substitutions for any matched medical jargon terms
    for term, plain in JARGON.items():
        if term.lower() not in [t.lower() for t in changed_terms]:
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            revised, count = re.subn(pattern, plain, out, flags=re.I)
            if count:
                out = revised
                changed_terms.append(term)

    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out).strip()
    if out.casefold() == sentence.strip().casefold():
        return {
            "suggestion": None,
            "status": "Human rewrite recommended",
            "note": "No safe template rewrite matched this sentence. Ask a clinical reviewer to rewrite it; do not substitute an unverified paraphrase.",
            "changed_terms": [],
        }
    return {
        "suggestion": out,
        "status": "Template-based draft suggestion",
        "note": "Verify the suggested wording against the source and clinical context before use. The rule-based rewrite is not clinically validated.",
        "changed_terms": changed_terms,
    }


def _is_non_patient_facing_metadata(sentence: str) -> bool:
    """Identify obvious administrative labels that should not be treated as patient-facing jargon."""
    s = sentence.strip().lower()
    if not s:
        return True
    if re.fullmatch(r"(?:unique\s+)?protocol\s+(?:id|number)\s*:?\s*", s):
        return True
    if re.fullmatch(r"(?:study|clinical trial)\s+(?:id|number|identifier)\s*:?\s*", s):
        return True
    if re.fullmatch(r"(?:version|document)\s+(?:number|date)\s*:?\s*", s):
        return True
    return False


def find_sentence_issues(
    text: str,
    target_grade: float = 8.0,
    nlp_predictions: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Identify sentence-level issues combining deterministic rules and optional ML predictions."""
    issues = []
    sentences = split_sentences(text)
    nlp_map = {p["sentence_number"]: p for p in (nlp_predictions or [])}

    for index, sentence in enumerate(sentences, start=1):
        sentence_words = words(sentence)
        if not sentence_words or _is_non_patient_facing_metadata(sentence):
            continue

        sentence_jargon = [term for term in JARGON if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", sentence, re.I)]
        sentence_flags = []

        if len(sentence_words) > 25:
            sentence_flags.append(f"Long sentence ({len(sentence_words)} words)")
        if sentence_jargon:
            sentence_flags.append("Medical jargon: " + ", ".join(sentence_jargon))

        matched_risk_categories = []
        for category, patterns in RISK_PATTERNS.items():
            if any(re.search(p, sentence, re.I) for p in patterns):
                matched_risk_categories.append(category)
        if matched_risk_categories:
            sentence_flags.append("Review risk wording: " + ", ".join(matched_risk_categories))

        # Check NLP model prediction for this sentence
        nlp_item = nlp_map.get(index)
        nlp_flag = False
        if nlp_item and nlp_item.get("is_flagged"):
            nlp_flag = True
            it = nlp_item.get("predicted_issue_type", "needs_review")
            prob = nlp_item.get("review_probability", 0.0)
            sentence_flags.append(f"NLP Model flag: {it} (review prob {prob:.0%})")

        if sentence_flags or nlp_flag:
            rewrite = simplify_sentence(sentence)
            issue_record = {
                "sentence_number": index,
                "original": sentence,
                "word_count": len(sentence_words),
                "issues": sentence_flags,
                "nlp_prediction": nlp_item,
                **rewrite,
            }
            issues.append(issue_record)

    return issues[:40]


def analyze_consent(
    text: str,
    target_grade: float = 8.0,
    procedure_type: str = "Research study",
    reproductive_applicable: bool = False,
    off_label_applicable: bool = False,
    use_nlp_classifier: bool = True,
) -> dict[str, Any]:
    """Analyze consent text and return reproducible metrics plus explanatory evidence."""
    normalized = clean_text(text)
    sentences = split_sentences(normalized)
    if len(words(normalized)) < 5:
        raise ValueError("Please provide a consent document with at least five words.")
    read = calculate_readability(normalized)

    # Core checklist
    element_results: dict[str, dict[str, Any]] = {}
    for key, spec in CORE_ELEMENTS.items():
        result = detect_element(normalized, spec["patterns"])
        element_results[key] = {"label": spec["label"], **result, "applicable": True, "conditional": False}

    if reproductive_applicable:
        spec = CONDITIONAL_ELEMENTS["reproductive"]
        result = detect_element(normalized, spec["patterns"])
        element_results["reproductive"] = {"label": spec["label"], **result, "applicable": True, "conditional": True}
    else:
        element_results["reproductive"] = {
            "label": CONDITIONAL_ELEMENTS["reproductive"]["label"],
            "present": None, "evidence": [], "match_count": 0, "applicable": False, "conditional": True
        }

    if off_label_applicable or procedure_type == "Off-label drug use":
        spec = CONDITIONAL_ELEMENTS["off_label"]
        result = detect_element(normalized, spec["patterns"])
        element_results["off_label"] = {"label": spec["label"], **result, "applicable": True, "conditional": True}
    else:
        element_results["off_label"] = {
            "label": CONDITIONAL_ELEMENTS["off_label"]["label"],
            "present": None, "evidence": [], "match_count": 0, "applicable": False, "conditional": True
        }

    applicable = [v for v in element_results.values() if v["applicable"]]
    missing_keys = [k for k, v in element_results.items() if v["applicable"] and not v["present"]]
    present_count = sum(bool(v["present"]) for v in applicable)
    structural_score = round(100 * present_count / max(1, len(applicable)), 1)

    jargon = detect_jargon(normalized)
    risk_language = detect_risk_language(normalized)
    uncertain_or_legal = [x for x in risk_language if x["category"] != "Strong risk disclosure (review for clarity)"]
    strong_risk_disclosure = [x for x in risk_language if x["category"] == "Strong risk disclosure (review for clarity)"]

    # Supervised NLP Classifier inference
    nlp_analysis: dict[str, Any] = {}
    if use_nlp_classifier:
        try:
            clf = SemanticConsentClassifier.get_default_instance()
            nlp_analysis = clf.score_document(sentences)
        except Exception:
            nlp_analysis = {"classifier_active": False, "reason": "NLP classifier failed to initialize."}
    else:
        nlp_analysis = {"classifier_active": False, "reason": "NLP classifier disabled by user setting."}

    # Semantic Risk Index computation
    grade_penalty = min(30.0, max(0.0, (read["flesch_kincaid_grade"] - target_grade) * 4.0))
    long_sentence_penalty = min(15.0, read["long_sentence_share"] * 15.0)
    jargon_count = sum(item["count"] for item in jargon)
    jargon_penalty = min(20.0, len(jargon) * 2.5 + max(0, jargon_count - len(jargon)) * 0.5)
    uncertainty_count = sum(item["count"] for item in uncertain_or_legal)
    uncertainty_penalty = min(15.0, uncertainty_count * 3.0)
    missing_core_count = sum(1 for key in missing_keys if key in CORE_ELEMENTS)
    missing_penalty = min(25.0, missing_core_count * 5.0)
    critical_missing_penalty = (5.0 if "autonomy" in missing_keys else 0.0) + (5.0 if "alternatives" in missing_keys else 0.0)

    raw_sri = grade_penalty + long_sentence_penalty + jargon_penalty + uncertainty_penalty + missing_penalty + critical_missing_penalty
    sri = int(round(min(100.0, max(0.0, raw_sri))))

    # Review priority accounts for critical checklist gaps and readability
    critical_gap = ("autonomy" in missing_keys) or ("alternatives" in missing_keys)
    if sri >= 60 or critical_gap or missing_core_count >= 3 or read["flesch_kincaid_grade"] > target_grade + 2:
        status = "RED"
        status_text = "High review priority: critical checklist gaps, high SRI, or substantially elevated reading grade detected."
    elif sri >= 30 or missing_core_count > 0 or read["flesch_kincaid_grade"] > target_grade:
        status = "YELLOW"
        status_text = "Review recommended: inspect readability, checklist evidence, and flagged phrases."
    else:
        status = "GREEN"
        status_text = "No major rule-based trigger found; human review is still required."

    if read["flesch_kincaid_grade"] <= target_grade:
        readability_status = "At or below selected target"
    elif read["flesch_kincaid_grade"] <= target_grade + 2:
        readability_status = "Slightly above selected target"
    else:
        readability_status = "Above selected target"

    # Sentence-level issues combined with ML predictions
    sentence_predictions = nlp_analysis.get("sentence_predictions", [])
    issues = find_sentence_issues(normalized, target_grade, sentence_predictions)

    return {
        "app_version": APP_VERSION,
        "procedure_type": procedure_type,
        "target_grade": float(target_grade),
        "word_count": read["word_count"],
        "sentence_count": read["sentence_count"],
        "average_sentence_length": read["average_sentence_length"],
        "long_sentence_count": read["long_sentence_count"],
        "long_sentence_share": read["long_sentence_share"],
        "flesch_kincaid_grade": read["flesch_kincaid_grade"],
        "gunning_fog_index": read["gunning_fog_index"],
        "flesch_reading_ease": read["flesch_reading_ease"],
        "automated_readability_index": read.get("automated_readability_index", 0.0),
        "complex_word_count": read["complex_word_count"],
        "readability_method": read["readability_method"],
        "readability_warning": read["readability_warning"],
        "readability_status": readability_status,
        "elements": element_results,
        "applicable_element_count": len(applicable),
        "present_element_count": present_count,
        "missing_elements": missing_keys,
        "structural_completeness_score": structural_score,
        "jargon": jargon,
        "jargon_unique_count": len(jargon),
        "jargon_occurrence_count": jargon_count,
        "risk_language": risk_language,
        "uncertainty_or_legal_flags": uncertain_or_legal,
        "strong_risk_disclosures": strong_risk_disclosure,
        "sentence_issues": issues,
        "nlp_analysis": nlp_analysis,
        "semantic_risk_index": sri,
        "semantic_risk_status": status,
        "semantic_risk_status_text": status_text,
        "sri_components": {
            "Readability above target (max 30)": round(grade_penalty, 2),
            "Long sentences (max 15)": round(long_sentence_penalty, 2),
            "Medical jargon (max 20)": round(jargon_penalty, 2),
            "Uncertainty / legalistic wording (max 15)": round(uncertainty_penalty, 2),
            "Missing applicable core elements (max 25)": round(missing_penalty, 2),
            "Missing autonomy / alternatives (additional)": round(critical_missing_penalty, 2),
        },
        "methodology_note": (
            "The Semantic Risk Index is a custom, deterministic heuristic for triaging documents. "
            "It is paired with a supervised TF-IDF NLP classifier for empirical sentence review. "
            "It is not a standardized clinical score and does not estimate an individual's probability of misunderstanding. "
            "A high-risk disclosure is flagged for clarity review but does not itself add to the SRI."
        ),
    }


def summary_record(result: dict[str, Any], document_name: str = "pasted_text") -> dict[str, Any]:
    nlp_info = result.get("nlp_analysis", {})
    record = {
        "document_name": document_name,
        "procedure_type": result["procedure_type"],
        "target_grade": result["target_grade"],
        "word_count": result["word_count"],
        "sentence_count": result["sentence_count"],
        "average_sentence_length": result["average_sentence_length"],
        "flesch_kincaid_grade": result["flesch_kincaid_grade"],
        "gunning_fog_index": result["gunning_fog_index"],
        "flesch_reading_ease": result["flesch_reading_ease"],
        "automated_readability_index": result.get("automated_readability_index", 0.0),
        "structural_completeness_score": result["structural_completeness_score"],
        "semantic_risk_index": result["semantic_risk_index"],
        "semantic_risk_status": result["semantic_risk_status"],
        "nlp_classifier_active": nlp_info.get("classifier_active", False),
        "nlp_flagged_share": nlp_info.get("flagged_share", 0.0),
        "nlp_mean_review_prob": nlp_info.get("mean_review_probability", 0.0),
        "nlp_flagged_count": nlp_info.get("flagged_count", 0),
        "jargon_unique_count": result["jargon_unique_count"],
        "jargon_occurrence_count": result["jargon_occurrence_count"],
        "missing_elements": "; ".join(result["missing_elements"]),
    }
    for key, item in result["elements"].items():
        record[f"element_{key}"] = "not_applicable" if not item["applicable"] else ("present_cue_detected" if item["present"] else "no_cue_detected")
    return record
