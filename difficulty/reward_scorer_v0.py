# difficulty/reward_scorer_v0.py
from difficulty.scorer_v0 import score_question as score_question_v0

def score(ex: dict) -> float:
    feats, s, bucket = score_question_v0(ex.get("problem_text", ""))
    return float(s)