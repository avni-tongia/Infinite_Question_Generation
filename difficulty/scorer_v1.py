from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from difficulty.concept_graph import detect_concepts, concept_graph_features
from difficulty.solver_features import solver_feature_proxies
from difficulty.utils import norm


@dataclass
class DifficultyWeightsV1:
    # Tune later
    concept_w: float = 0.25
    cross_w: float = 0.25
    depth_w: float = 0.10
    solver_w: float = 0.25
    eq_w: float = 0.10
    ambiguity_w: float = 0.05


def ambiguity_proxy(sfeat: Dict[str, float]) -> float:
    """
    Higher ambiguity => harder:
    - fewer numeric givens (nums) relative to variables (vars) makes abstraction harder
    """
    vars_ = float(sfeat.get("variable_count_proxy", 0.0))
    nums_ = float(sfeat.get("numeric_givens_proxy", 0.0))

    if vars_ <= 0:
        return 0.0

    ratio = (vars_ + 1.0) / (nums_ + 1.0)
    return max(0.0, ratio - 1.0)


def extract_features_v1(
    question: str,
    equations: Optional[List[str]] = None,
    solver_trace: Optional[Dict] = None,
) -> Dict[str, float]:
    concepts = detect_concepts(question)
    gfeat = concept_graph_features(concepts)
    sfeat = solver_feature_proxies(question, equations=equations, solver_trace=solver_trace)

    feats: Dict[str, float] = {}
    feats.update(gfeat)
    feats.update(sfeat)

    feats["ambiguity_proxy"] = float(ambiguity_proxy(sfeat))
    feats["concepts_detected"] = float(len(concepts))  # redundant but useful
    return feats


def score_v1(
    feats: Dict[str, float],
    w: DifficultyWeightsV1 = DifficultyWeightsV1(),
) -> float:
    """
    Weighted normalized score (v1).
    Normalizations chosen so each term is ~1-ish for typical outputs.
    """
    concept_n = norm(feats["concept_count"], 3.0)                 # 3 concepts ~1
    cross_n = norm(feats["cross_chapter_score"], 2.0)             # 2 cross ~1
    depth_n = norm(feats["graph_depth_proxy"], 3.0)               # 3 depth ~1
    solver_n = norm(feats["solver_steps_proxy"], 10.0)            # 10 steps ~1
    eq_n = norm(feats["equation_count_proxy"], 3.0)               # 3 eq ~1
    amb_n = norm(feats["ambiguity_proxy"], 1.0)                   # already ~0..1+

    return (
        w.concept_w * concept_n
        + w.cross_w * cross_n
        + w.depth_w * depth_n
        + w.solver_w * solver_n
        + w.eq_w * eq_n
        + w.ambiguity_w * amb_n
    )


def bucket_v1(score_value: float) -> str:
    """
    Initial thresholds (v1). Tune after seeing histograms.
    """
    if score_value < 0.70:
        return "easy"
    if score_value < 1.25:
        return "medium"
    return "hard"


def score_question_v1(
    question: str,
    equations: Optional[List[str]] = None,
    solver_trace: Optional[Dict] = None,
) -> Tuple[Dict[str, float], float, str]:
    """
    ✅ Exported function expected by difficulty/scorer.py dispatcher.
    """
    feats = extract_features_v1(question, equations=equations, solver_trace=solver_trace)
    s = float(score_v1(feats))
    b = bucket_v1(s)
    return feats, s, b