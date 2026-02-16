from __future__ import annotations

import re
from typing import Dict, List, Optional

_VAR_RE = re.compile(r"\b[a-zA-Z]\b")  # single-letter variable proxy
_NUM_RE = re.compile(r"\d+(\.\d+)?")   # number proxy


def solver_feature_proxies(
    question: str,
    equations: Optional[List[str]] = None,
    solver_trace: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    v1: cheap proxies for solver complexity.
    Later: replace with real solver stats (steps, substitutions, eliminations).
    """
    q = question.strip()

    # Equation count proxy: from extracted equations if provided, else '=' in text
    if equations is not None:
        eq_count = len([e for e in equations if str(e).strip()])
    else:
        eq_count = q.count("=")

    # Variable count proxy (single-letter vars appear often in physics)
    var_count = len(_VAR_RE.findall(q))

    # Numeric givens proxy: more numbers often helps reduce ambiguity
    num_count = len(_NUM_RE.findall(q))

    # If you have a real trace, use it
    steps = 0.0
    subs = 0.0
    if solver_trace:
        steps = float(solver_trace.get("steps", 0) or 0)
        subs = float(solver_trace.get("substitutions", 0) or 0)

    # If no real steps, approximate with structure
    if steps == 0.0:
        # heuristic: more equations + more vars => more work
        steps = float(eq_count * 2) + float(var_count / 5.0)

    return {
        "equation_count_proxy": float(eq_count),
        "variable_count_proxy": float(var_count),
        "numeric_givens_proxy": float(num_count),
        "solver_steps_proxy": float(steps),
        "substitutions_proxy": float(subs),
    }