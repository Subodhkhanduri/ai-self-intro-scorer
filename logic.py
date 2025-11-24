"""
logic.py - Scoring engine for AI Communication Coach (Self-introduction rubric)

Implements the rubric defined in the provided case-study spreadsheet.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Tuple

# Third-party NLP libraries
try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    st_util = None  # type: ignore

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore

try:
    import language_tool_python
except ImportError:  # pragma: no cover
    language_tool_python = None  # type: ignore


# ---------------------------------------------------------------------------
# Optional helper to install dependencies (to be run in your environment)
# ---------------------------------------------------------------------------

def install_dependencies():
    """
    Helper function to install required dependencies.

    Note:
        This uses pip and assumes it is available in the runtime environment.
        Call this manually if you need a one-shot installer.

    Example:
        >>> from logic import install_dependencies
        >>> install_dependencies()
    """
    import subprocess
    import sys

    packages = [
        "spacy",
        "sentence-transformers",
        "vaderSentiment",
        "language-tool-python",
    ]

    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    # Optional: download a small English model for spaCy
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except Exception:
        # If it fails (no internet, etc.), spaCy will fall back to a blank model.
        pass


# ---------------------------------------------------------------------------
# Global, lazily initialised NLP objects
# ---------------------------------------------------------------------------

_NLP = None
_ST_MODEL = None
_VADER = None
_LT_TOOL = None


def get_nlp():
    """Return a spaCy English pipeline, falling back to a blank model if needed."""
    global _NLP
    if _NLP is None:
        if spacy is None:
            raise ImportError("spaCy is not installed. Run install_dependencies().")
        try:
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            # Fallback: tokenizer-only blank model
            _NLP = spacy.blank("en")
    return _NLP


def get_sentence_transformer():
    """Return a SentenceTransformer model, loaded lazily."""
    global _ST_MODEL
    if _ST_MODEL is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Run install_dependencies().")
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def get_vader():
    """Return VADER sentiment analyzer."""
    global _VADER
    if _VADER is None:
        if SentimentIntensityAnalyzer is None:
            raise ImportError("vaderSentiment is not installed. Run install_dependencies().")
        _VADER = SentimentIntensityAnalyzer()
    return _VADER


def get_language_tool():
    """Return LanguageTool instance for English grammar checking."""
    global _LT_TOOL
    if _LT_TOOL is None:
        if language_tool_python is None:
            raise ImportError("language-tool-python is not installed. Run install_dependencies().")
        _LT_TOOL = language_tool_python.LanguageTool("en-US")
    return _LT_TOOL


# ---------------------------------------------------------------------------
# Utility / preprocessing helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Collapse extra whitespace and strip."""
    return " ".join((text or "").strip().split())


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into word-like tokens using spaCy if available,
    otherwise a simple regex fallback.
    """
    nlp = get_nlp()
    try:
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_space]
    except Exception:
        tokens = re.findall(r"\b\w+\b", text)
    return tokens


def split_sentences(text: str) -> List[str]:
    """Return a list of sentence strings using spaCy if possible."""
    nlp = get_nlp()
    try:
        doc = nlp(text)
        # If the pipeline has no sentence segmenter, fall back to simple split.
        if not list(doc.sents):
            raise ValueError
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    except Exception:
        # Simple rule-based sentence split as fallback
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def count_words(text: str) -> int:
    """Convenience wrapper."""
    return len(tokenize_words(text))


# ---------------------------------------------------------------------------
# Content & Structure scoring
# ---------------------------------------------------------------------------

SALUTATION_NORMAL = ["hi", "hello"]
SALUTATION_GOOD = [
    "good morning",
    "good afternoon",
    "good evening",
    "good day",
    "hello everyone",
]
SALUTATION_EXCELLENT_PHRASES = [
    "excited to be here",
    "excited to introduce",
    "i am excited to introduce",
    "feeling great",
    "great to be here",
    "happy to be here",
]

# Must-have and good-to-have keyword groups from rubric
MUST_HAVE_KEYWORDS = {
    "name": ["my name is", "i am", "i'm", "this is"],
    "age": ["years old", "year old", "my age is"],
    "school_class": ["school", "class", "standard", "grade"],
    "family": ["family", "my parents", "my father", "my mother", "brother", "sister"],
    "hobbies": ["hobby", "hobbies", "like to do", "love to", "enjoy", "in my free time"],
}

GOOD_TO_HAVE_KEYWORDS = {
    "origin": ["i am from", "i'm from", "i come from", "my hometown"],
    "ambition": ["my dream is", "my goal is", "i want to become", "i want to be"],
    "fun_fact": ["fun fact", "interesting thing", "something unique"],
    "strengths": ["my strength", "my strengths", "i am good at", "i'm good at"],
    "achievements": ["i have won", "i have achieved", "my achievement", "my achievements"],
}


def _semantic_keyword_match(text: str, keyword_variants: List[str], threshold: float = 0.6) -> bool:
    """
    Use sentence-transformers to semantically match any keyword variant
    against the transcript text (at sentence level).

    This is used as a fallback when literal matching fails, to make the
    rubric a bit more robust to paraphrases.
    """
    if SentenceTransformer is None or st_util is None:
        return False

    sentences = split_sentences(text)
    if not sentences:
        return False

    # Quick literal check first.
    tl = text.lower()
    for kv in keyword_variants:
        if kv.lower() in tl:
            return True

    try:
        model = get_sentence_transformer()
    except Exception:
        return False

    sent_embeds = model.encode(sentences, convert_to_tensor=True)
    key_embeds = model.encode(keyword_variants, convert_to_tensor=True)

    sim_matrix = st_util.cos_sim(sent_embeds, key_embeds)
    max_sim = float(sim_matrix.max().item())
    return max_sim >= threshold


def score_salutation(text: str) -> Tuple[int, Dict[str, Any]]:
    """
    Salutation scoring per rubric:

    - Excellent: includes "I am excited to introduce / Feeling great / great to be here" -> 5
    - Good: "Good Morning/Afternoon/Evening/Day", "Hello everyone" -> 4
    - Normal: "Hi", "Hello" -> 2
    - No salutation -> 0
    """
    tl = text.lower()

    for phrase in SALUTATION_EXCELLENT_PHRASES:
        if phrase in tl:
            return 5, {"level": "excellent", "matched_phrase": phrase}

    for phrase in SALUTATION_GOOD:
        if phrase in tl:
            return 4, {"level": "good", "matched_phrase": phrase}

    for phrase in SALUTATION_NORMAL:
        # Avoid matching "hello" inside other words, but keep things simple.
        if re.search(r"\b" + re.escape(phrase) + r"\b", tl):
            return 2, {"level": "normal", "matched_phrase": phrase}

    return 0, {"level": "none", "matched_phrase": None}


def score_keywords(text: str) -> Tuple[int, Dict[str, Any]]:
    """
    Keyword presence scoring.

    Must-have (each 4 pts, max 20):
        - Name
        - Age
        - School/Class
        - Family
        - Hobbies / Interests

    Good-to-have (each 2 pts, max 10):
        - Origin location
        - Ambition / Goal / Dream
        - Fun fact / Something unique
        - Strengths
        - Achievements
    """
    tl = text.lower()
    must_have_results = {}
    good_to_have_results = {}

    must_have_score = 0
    for key, phrases in MUST_HAVE_KEYWORDS.items():
        present = any(p.lower() in tl for p in phrases)
        # If not literally present, try semantic matching
        if not present:
            try:
                present = _semantic_keyword_match(text, phrases)
            except Exception:
                present = False
        must_have_results[key] = present
        if present:
            must_have_score += 4

    good_to_have_score = 0
    for key, phrases in GOOD_TO_HAVE_KEYWORDS.items():
        present = any(p.lower() in tl for p in phrases)
        if not present:
            try:
                present = _semantic_keyword_match(text, phrases)
            except Exception:
                present = False
        good_to_have_results[key] = present
        if present:
            good_to_have_score += 2

    total_score = min(must_have_score + good_to_have_score, 30)

    feedback = {
        "must_have_present": [k for k, v in must_have_results.items() if v],
        "must_have_missing": [k for k, v in must_have_results.items() if not v],
        "good_to_have_present": [k for k, v in good_to_have_results.items() if v],
        "good_to_have_missing": [k for k, v in good_to_have_results.items() if not v],
    }

    return total_score, feedback


CLOSING_PHRASES = [
    "thank you",
    "thanks for listening",
    "thanks for your time",
    "that's all",
    "that is all",
    "nice to meet you",
]


def score_flow(text: str) -> Tuple[int, Dict[str, Any]]:
    """
    Flow scoring: Order should be

        Salutation -> Basic details (Name etc.) -> Additional details -> Closing

    We'll approximate with index-based checks on the raw text.
    """
    tl = text.lower()

    def find_first(phrases: List[str]) -> Optional[int]:
        positions = [tl.find(p) for p in phrases if tl.find(p) != -1]
        return min(positions) if positions else None

    sal_pos = find_first(SALUTATION_NORMAL + SALUTATION_GOOD + SALUTATION_EXCELLENT_PHRASES)

    basic_detail_markers: List[str] = []
    for phrases in MUST_HAVE_KEYWORDS.values():
        basic_detail_markers.extend(phrases)
    basic_pos = find_first(basic_detail_markers)

    additional_detail_markers: List[str] = []
    for phrases in GOOD_TO_HAVE_KEYWORDS.values():
        additional_detail_markers.extend(phrases)
    add_pos = find_first(additional_detail_markers)

    close_pos = find_first(CLOSING_PHRASES)

    positions = [
        ("salutation", sal_pos),
        ("basic", basic_pos),
        ("additional", add_pos),
        ("closing", close_pos),
    ]
    # Keep only those that exist
    existing = [(label, pos) for label, pos in positions if pos is not None]

    order_followed = False
    if len(existing) >= 2:
        # Check that positions are strictly increasing
        pos_values = [pos for _, pos in existing]
        order_followed = all(earlier < later for earlier, later in zip(pos_values, pos_values[1:]))

    return (5 if order_followed else 0), {
        "order_followed": order_followed,
        "positions": {label: pos for label, pos in positions},
    }


# ---------------------------------------------------------------------------
# Speech rate scoring
# ---------------------------------------------------------------------------

def compute_wpm(word_count: int, audio_duration_seconds: float) -> float:
    """Compute words per minute from word count and audio duration."""
    if audio_duration_seconds <= 0:
        return 0.0
    return (word_count / audio_duration_seconds) * 60.0


def score_speech_rate(wpm: float) -> Tuple[int, Dict[str, Any]]:
    """
    Rubric mapping:

        >=161 WPM       -> 2 (Too fast)
        141–160 WPM     -> 6 (Fast)
        111–140 WPM     -> 10 (Ideal)
        81–110 WPM      -> 6 (Slow)
        <80 WPM         -> 2 (Too slow)
    """
    if wpm >= 161:
        score = 2
        band = "too_fast"
    elif 141 <= wpm <= 160:
        score = 6
        band = "fast"
    elif 111 <= wpm <= 140:
        score = 10
        band = "ideal"
    elif 81 <= wpm <= 110:
        score = 6
        band = "slow"
    else:  # wpm < 80
        score = 2
        band = "too_slow"

    return score, {"band": band, "wpm": wpm}


# ---------------------------------------------------------------------------
# Language & Grammar scoring
# ---------------------------------------------------------------------------

def analyze_grammar(text: str, total_words: int) -> Tuple[int, float, List[Dict[str, Any]]]:
    """
    Use language-tool-python to count grammar errors,
    compute errors per 100 words and return a 0–10 score + error list.

    Formula in rubric:
        fraction = 1 - min(errors_per_100_words / 10, 1)
        Then map fraction to discrete bands:
            >0.9        -> 10
            0.7–0.89    -> 8
            0.5–0.69    -> 6
            0.3–0.49    -> 4
            0–0.29      -> 2
    """
    if total_words == 0:
        return 0, 0.0, []

    try:
        tool = get_language_tool()
    except Exception:
        # If grammar tool not available, skip scoring but avoid crash
        return 0, 0.0, []

    matches = tool.check(text)
    error_count = len(matches)
    errors_per_100 = (error_count / total_words) * 100.0
    fraction = 1.0 - min(errors_per_100 / 10.0, 1.0)

    if fraction > 0.9:
        score = 10
    elif fraction >= 0.7:
        score = 8
    elif fraction >= 0.5:
        score = 6
    elif fraction >= 0.3:
        score = 4
    else:
        score = 2

    errors_detail: List[Dict[str, Any]] = []
    for m in matches:
        errors_detail.append(
            {
                "message": m.message,
                "offset": m.offset,
                "error_length": m.errorLength,
                "context": m.context,
                "rule_id": m.ruleId,
            }
        )

    return score, fraction, errors_detail


def score_vocabulary_ttr(words: List[str]) -> Tuple[int, float]:
    """
    Vocabulary richness via Type-Token Ratio (TTR).

    Rubric mapping:

        TTR 0.9–1.0     -> 10
        0.7–0.89        -> 8
        0.5–0.69        -> 6
        0.3–0.49        -> 4
        0–0.29          -> 2
    """
    if not words:
        return 0, 0.0

    # Lowercase tokens for uniqueness
    uniq = set(w.lower() for w in words)
    ttr = len(uniq) / float(len(words))

    if ttr >= 0.9:
        score = 10
    elif ttr >= 0.7:
        score = 8
    elif ttr >= 0.5:
        score = 6
    elif ttr >= 0.3:
        score = 4
    else:
        score = 2

    return score, ttr


# ---------------------------------------------------------------------------
# Clarity scoring (filler words)
# ---------------------------------------------------------------------------

FILLER_WORDS = ["um", "uh", "like", "you know", "actually", "basically"]


def score_clarity_filler(text: str, words: List[str]) -> Tuple[int, Dict[str, Any]]:
    """
    Filler word scoring.

    Rate = (filler_count / total_words) * 100

    Rubric mapping:

        0–3%           -> 15
        4–6%           -> 12
        7–9%           -> 9
        10–12%         -> 6
        >=13%          -> 3
    """
    total_words = len(words)
    if total_words == 0:
        return 0, {"rate": 0.0, "filler_count": 0, "filler_words": []}

    tl = text.lower()
    filler_count = 0
    found_fillers: List[str] = []

    # Count fillers by regex in the full text
    for fw in FILLER_WORDS:
        pattern = r"\b" + re.escape(fw) + r"\b"
        matches = re.findall(pattern, tl)
        if matches:
            filler_count += len(matches)
            found_fillers.extend([fw] * len(matches))

    rate = (filler_count / total_words) * 100.0

    if rate <= 3:
        score = 15
    elif rate <= 6:
        score = 12
    elif rate <= 9:
        score = 9
    elif rate <= 12:
        score = 6
    else:
        score = 3

    return score, {
        "rate": rate,
        "filler_count": filler_count,
        "filler_words": found_fillers,
    }


# ---------------------------------------------------------------------------
# Engagement scoring (VADER sentiment)
# ---------------------------------------------------------------------------

def score_engagement(text: str) -> Tuple[int, Dict[str, Any]]:
    """
    Engagement via VADER positive score.

    Rubric mapping (using VADER 'pos' score):

        >=0.9      -> 15
        0.7–0.89   -> 12
        0.5–0.69   -> 9
        0.3–0.49   -> 6
        <0.3       -> 3
    """
    if not text.strip():
        return 0, {"positive": 0.0, "compound": 0.0}

    try:
        vader = get_vader()
    except Exception:
        return 0, {"positive": 0.0, "compound": 0.0}

    scores = vader.polarity_scores(text)
    positive = scores.get("pos", 0.0)
    compound = scores.get("compound", 0.0)

    if positive >= 0.9:
        score = 15
    elif positive >= 0.7:
        score = 12
    elif positive >= 0.5:
        score = 9
    elif positive >= 0.3:
        score = 6
    else:
        score = 3

    return score, {"positive": positive, "compound": compound}


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_transcript(transcript_text: str, audio_duration_seconds: float) -> Dict[str, Any]:
    """
    Analyze a self-introduction transcript according to the provided rubric.

    Args:
        transcript_text: The raw transcript string.
        audio_duration_seconds: Duration of the audio in seconds.

    Returns:
        dict with keys:
            - overall_score (0–100)
            - criteria_scores (detailed breakdown)
            - feedback (diagnostic information)
    """
    transcript_text = normalize_text(transcript_text)

    # Basic counts
    words = tokenize_words(transcript_text)
    total_words = len(words)
    wpm = compute_wpm(total_words, audio_duration_seconds)

    # Content & Structure
    salutation_score, salutation_info = score_salutation(transcript_text)
    keyword_score, keyword_info = score_keywords(transcript_text)
    flow_score, flow_info = score_flow(transcript_text)
    content_structure_score = salutation_score + keyword_score + flow_score  # /40 max

    # Speech Rate
    speech_rate_score, speech_rate_info = score_speech_rate(wpm)

    # Language & Grammar
    grammar_score, grammar_fraction, grammar_errors = analyze_grammar(transcript_text, total_words)
    vocab_score, ttr = score_vocabulary_ttr(words)
    language_grammar_score = grammar_score + vocab_score  # /20 max

    # Clarity
    clarity_score, clarity_info = score_clarity_filler(transcript_text, words)

    # Engagement
    engagement_score, engagement_info = score_engagement(transcript_text)

    # Overall
    overall_score = (
        content_structure_score
        + speech_rate_score
        + language_grammar_score
        + clarity_score
        + engagement_score
    )

    # Assemble criteria breakdown
    criteria_scores: Dict[str, Any] = {
        "content_structure": {
            "score": content_structure_score,
            "max_score": 40,
            "details": {
                "salutation": {"score": salutation_score, "max_score": 5, **salutation_info},
                "keywords": {"score": keyword_score, "max_score": 30, **keyword_info},
                "flow": {"score": flow_score, "max_score": 5, **flow_info},
            },
        },
        "speech_rate": {
            "score": speech_rate_score,
            "max_score": 10,
            "details": speech_rate_info,
        },
        "language_grammar": {
            "score": language_grammar_score,
            "max_score": 20,
            "details": {
                "grammar": {
                    "score": grammar_score,
                    "max_score": 10,
                    "fraction": grammar_fraction,
                    "errors": grammar_errors,
                },
                "vocabulary": {
                    "score": vocab_score,
                    "max_score": 10,
                    "ttr": ttr,
                },
            },
        },
        "clarity": {
            "score": clarity_score,
            "max_score": 15,
            "details": clarity_info,
        },
        "engagement": {
            "score": engagement_score,
            "max_score": 15,
            "details": engagement_info,
        },
    }

    # Feedback aggregation
    feedback: Dict[str, Any] = {
        "missing_must_have_keywords": keyword_info.get("must_have_missing", []),
        "missing_good_to_have_keywords": keyword_info.get("good_to_have_missing", []),
        "grammar_errors": grammar_errors,
        "filler_words": clarity_info.get("filler_words", []),
        "speech_rate_band": speech_rate_info.get("band"),
        "wpm": wpm,
        "engagement_positive_score": engagement_info.get("positive", 0.0),
    }

    return {
        "overall_score": float(overall_score),
        "criteria_scores": criteria_scores,
        "feedback": feedback,
    }


if __name__ == "__main__":  # Simple manual test stub
    sample_text = (
        "Hello everyone, my name is Alex. I am 12 years old and I study in grade 7 at Greenfield School. "
        "I live with my parents and younger sister. In my free time, I love to play football and read mystery books. "
        "My dream is to become a scientist one day. Thank you for listening!"
    )
    result = analyze_transcript(sample_text, audio_duration_seconds=60)
    import json

    print(json.dumps(result, indent=2))
