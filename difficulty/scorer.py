from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from difficulty.scorer_v0 import score_question as score_question_v0
from difficulty.scorer_v1 import score_question_v1


def score_question(
    question: str,
    scorer: str = "v0",
    equations: Optional[List[str]] = None,
    solver_trace: Optional[Dict] = None,
) -> Tuple[Dict[str, float], float, str]:
    """
    Unified difficulty scoring API.

    scorer:
      - "v0": old heuristic scorer (text-only)
      - "v1": concept-graph + solver proxies (text-only for now, optional equations/trace)
    """
    scorer = (scorer or "v0").lower().strip()

    if scorer == "v1":
        return score_question_v1(question, equations=equations, solver_trace=solver_trace)

    # default to v0
    return score_question_v0(question)