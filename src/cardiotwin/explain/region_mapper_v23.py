from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

REGION_TO_LEADS = {
    "anterior": ["V2", "V3", "V4"],
    "septal": ["V1", "V2"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "inferior": ["II", "III", "aVF"],
    "global_conduction": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "hypertrophy_chamber": ["I", "aVL", "V1", "V5", "V6"],
}

CLASS_REGION_PRIOR = {
    "NORM": {},
    "MI": {
        "anterior": 1.15,
        "inferior": 1.10,
        "lateral": 1.05,
        "septal": 1.00,
    },
    "STTC": {
        "anterior": 1.10,
        "inferior": 1.10,
        "lateral": 1.05,
        "septal": 0.95,
    },
    "CD": {
        "global_conduction": 1.35,
        "septal": 1.05,
        "anterior": 0.90,
        "lateral": 0.85,
        "inferior": 0.85,
    },
    "HYP": {
        "hypertrophy_chamber": 1.35,
        "lateral": 0.95,
        "anterior": 0.90,
        "inferior": 0.75,
    },
}

DEFAULT_REGION_ORDER = [
    "anterior",
    "septal",
    "inferior",
    "lateral",
    "global_conduction",
    "hypertrophy_chamber",
]


def normalize_lead_evidence(lead_scores: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in lead_scores.items():
        try:
            x = float(v)
        except Exception:
            x = 0.0
        if math.isnan(x) or math.isinf(x):
            x = 0.0
        out[k] = max(0.0, x)
    total = sum(out.values())
    if total <= 0:
        return out
    return {k: v / total for k, v in out.items()}


def region_scores_from_leads(
    lead_scores: Dict[str, float],
    predicted_class: str,
    class_probability: float = 1.0,
) -> Dict[str, float]:
    """
    Lateral-bias fix:
    1. Normalize evidence by number of leads in each region.
    2. Apply class-region prior.
    3. Do not let lateral win only because it has more leads.
    """
    lead_scores = normalize_lead_evidence(lead_scores)
    priors = CLASS_REGION_PRIOR.get(predicted_class, {})

    scores = {}
    for region, leads in REGION_TO_LEADS.items():
        vals = [lead_scores.get(l, 0.0) for l in leads]
        # mean instead of sum = core lateral-bias fix
        base = sum(vals) / max(len(vals), 1)
        prior = priors.get(region, 1.0)
        scores[region] = float(base * prior * float(class_probability))

    return scores


def decide_region(
    region_scores: Dict[str, float],
    min_evidence: float = 0.015,
    uncertainty_margin: float = 0.20,
) -> Dict[str, object]:
    """
    Returns uncertain when:
    - total/top evidence is too low
    - top and second region are too close
    """
    items = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
    if not items:
        return {
            "region": "uncertain",
            "confidence": 0.0,
            "reason": "no_region_scores",
            "ranked_regions": [],
        }

    top_region, top_score = items[0]
    second_region, second_score = items[1] if len(items) > 1 else ("none", 0.0)

    if top_score < min_evidence:
        return {
            "region": "uncertain",
            "confidence": float(top_score),
            "reason": "low_total_region_evidence",
            "ranked_regions": items,
        }

    margin = (top_score - second_score) / max(top_score, 1e-12)
    if margin < uncertainty_margin:
        return {
            "region": "uncertain",
            "confidence": float(top_score),
            "reason": "top_region_margin_too_small",
            "top_region": top_region,
            "second_region": second_region,
            "margin": float(margin),
            "ranked_regions": items,
        }

    return {
        "region": top_region,
        "confidence": float(top_score),
        "reason": "accepted",
        "top_region": top_region,
        "second_region": second_region,
        "margin": float(margin),
        "ranked_regions": items,
    }


def map_prediction_to_region(
    lead_scores: Dict[str, float],
    predicted_class: str,
    class_probability: float,
) -> Dict[str, object]:
    scores = region_scores_from_leads(
        lead_scores=lead_scores,
        predicted_class=predicted_class,
        class_probability=class_probability,
    )
    decision = decide_region(scores)
    decision["predicted_class"] = predicted_class
    decision["class_probability"] = float(class_probability)
    decision["region_scores"] = scores
    return decision
