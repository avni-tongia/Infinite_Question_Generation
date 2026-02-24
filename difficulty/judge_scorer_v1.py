# difficulty/judge_scorer_v1.py
from difficulty.scorer_v1 import score_question_v1

def score(ex: dict) -> float:
    feats, s, bucket = score_question_v1(
        ex.get("problem_text", ""),
        equations=None,
        solver_trace=None
    )
    return float(s)