"""
Difficulty scorer (v0): interpretable features.

This is intentionally simple at first:
- We start with robust, cheap proxies.
- Later, we can plug in equation tagging, concept graphs, etc.

Outputs:
- features dict
- scalar score (weighted sum)
- bucket (easy/medium/hard) by score thresholds
"""

import re
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class DifficultyWeights:
    # You will tune these later using ablations
    length_w: float = 0.10
    symbol_w: float = 0.20
    step_w: float = 0.35
    algebra_w: float = 0.35


def _count_math_symbols(text: str) -> int:
    """Count basic math symbols as a proxy for equation density."""
    return sum(text.count(ch) for ch in ["=", "+", "-", "*", "/", "^"])


def _step_proxy(text: str) -> int:
    """
    Very rough proxy for multi-step structure.
    Heuristic: count of cue words that often indicate multi-step reasoning.
    """
    cues = ["then", "therefore", "hence", "after", "before", "finally", "find", "calculate"]
    t = text.lower()
    return sum(t.count(c) for c in cues)


def _algebra_proxy(text: str) -> int:
    """
    Proxy for algebraic complexity:
    - detect words indicating quadratic/simultaneous/trig style.
    """
    t = text.lower()
    score = 0
    if any(w in t for w in ["simultaneous", "system of", "two equations"]):
        score += 2
    if any(w in t for w in ["quadratic", "square", "root", "sqrt"]):
        score += 2
    if any(w in t for w in ["sin", "cos", "tan", "theta", "angle"]):
        score += 1
    return score


def extract_features(question: str) -> Dict[str, float]:
    """Extract a small set of interpretable difficulty features."""
    q = question.strip()
    length = len(q.split())
    symbols = _count_math_symbols(q)
    step = _step_proxy(q)
    algebra = _algebra_proxy(q)

    return {
        "length_words": float(length),
        "math_symbol_count": float(symbols),
        "step_proxy": float(step),
        "algebra_proxy": float(algebra),
    }


def score(features: Dict[str, float], w: DifficultyWeights = DifficultyWeights()) -> float:
    """
    Compute scalar difficulty score.
    Normalize lightly to keep magnitudes comparable.
    """
    length_n = features["length_words"] / 60.0         # ~1 around 60 words
    symbols_n = features["math_symbol_count"] / 8.0     # ~1 around 8 symbols
    step_n = features["step_proxy"] / 6.0               # ~1 around 6 cues
    algebra_n = features["algebra_proxy"] / 4.0         # ~1 around 4 score

    return (
        w.length_w * length_n +
        w.symbol_w * symbols_n +
        w.step_w * step_n +
        w.algebra_w * algebra_n
    )


def bucket(score_value: float) -> str:
    """
    Convert scalar score -> {easy, medium, hard}.
    These thresholds are initial and will be tuned after we see baseline histograms.
    """
    if score_value < 0.55:
        return "easy"
    if score_value < 1.05:
        return "medium"
    return "hard"


def score_question(question: str) -> Tuple[Dict[str, float], float, str]:
    """
    ✅ IMPORTANT: This is the exported function the dispatcher expects.
    """
    feats = extract_features(question)
    s = score(feats)
    b = bucket(s)
    return feats, s, b