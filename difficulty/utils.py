from __future__ import annotations

from typing import Optional


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def norm(x: float, denom: float) -> float:
    """Light normalization: x/denom with denom>0."""
    return safe_div(float(x), float(denom), default=0.0)