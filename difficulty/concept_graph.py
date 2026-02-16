from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

# NOTE:
# This is intentionally lightweight + interpretable.
# You will later replace/augment this with:
# - equation-to-chapter tagging (from extracted equations)
# - concept-distance graph (global chapter graph)
# - variable-bridging across concept equations

CHAPTER_KEYWORDS: Dict[str, List[str]] = {
    "kinematics": [
        "displacement", "distance", "velocity", "speed", "acceleration", "time",
        "u", "v", "a", "s", "t", "uniform", "retardation"
    ],
    "newton_laws": [
        "force", "friction", "normal", "tension", "mass", "newton", "f = ma",
        "coefficient", "mu", "reaction", "pulley", "string"
    ],
    "work_energy": [
        "work", "energy", "kinetic", "potential", "conservation of energy",
        "power", "spring", "k x", "compression"
    ],
    "momentum": [
        "momentum", "impulse", "collision", "inelastic", "elastic", "recoil",
        "center of mass"
    ],
    "circular_motion": [
        "circular", "centripetal", "angular", "omega", "rad/s", "v^2/r",
        "banking", "radius", "revolution"
    ],
    "gravitation": [
        "gravity", "gravitational", "g", "planet", "orbital", "escape velocity",
        "universal gravitation"
    ],
}

_WORD_RE = re.compile(r"[a-zA-Z]+")


def detect_concepts(text: str) -> Set[str]:
    """
    Very lightweight concept detection.
    Returns a set of concept keys (chapter-like buckets).
    """
    t = text.lower()
    concepts: Set[str] = set()

    # token-ish list helps reduce accidental substring matches a bit
    words = set(_WORD_RE.findall(t))

    for concept, keys in CHAPTER_KEYWORDS.items():
        hit = False
        for k in keys:
            k_low = k.lower()

            # If keyword looks like a phrase, do substring search.
            if " " in k_low or "=" in k_low or "^" in k_low or "/" in k_low:
                if k_low in t:
                    hit = True
                    break
            else:
                # single word -> token match
                if k_low in words:
                    hit = True
                    break

        if hit:
            concepts.add(concept)

    return concepts


def concept_graph_features(concepts: Set[str]) -> Dict[str, float]:
    """
    Graph-ish features derived only from concept set for now.

    Later upgrades:
    - edges via variable-bridging across equations
    - depth via ordered dependency extraction
    - distance via global chapter graph shortest-paths
    """
    c = len(concepts)

    # crude coupling: if multiple concepts appear, assume some coupling exists
    cross = max(0, c - 1)

    # depth proxy: more concepts often implies longer reasoning chain (coarse)
    depth = float(c)

    return {
        "concept_count": float(c),
        "cross_chapter_score": float(cross),
        "graph_depth_proxy": float(depth),
    }