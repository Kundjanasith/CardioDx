from __future__ import annotations
from cardiotwin.constants import LEAD_TO_REGION, REGION_TO_LEADS, CLASS_TO_REGION_PRIOR, REGIONS


def lead_to_region_scores(lead_importance: dict[str, float]) -> dict[str, float]:
    scores = {r: 0.0 for r in REGIONS}
    for lead, imp in lead_importance.items():
        region = LEAD_TO_REGION.get(lead)
        if region:
            scores[region] += float(max(imp, 0.0))
        else:
            scores["uncertain"] += float(max(imp, 0.0))
    total = sum(scores.values()) + 1e-9
    return {k: float(v / total) for k, v in scores.items()}


def class_prior_region_scores(class_probabilities: dict[str, float]) -> dict[str, float]:
    scores = {r: 0.0 for r in REGIONS}
    for cls, prob in class_probabilities.items():
        priors = CLASS_TO_REGION_PRIOR.get(cls, {})
        for region, w in priors.items():
            scores[region] += float(prob) * float(w)
    maxv = max(scores.values()) if scores else 0.0
    if maxv > 0:
        scores = {k: float(v / maxv) for k, v in scores.items()}
    return scores


def fuse_region_scores(lead_scores: dict[str, float], class_scores: dict[str, float], sqi: float = 1.0,
                       alpha: float = 0.60) -> dict[str, float]:
    regions = sorted(set(lead_scores) | set(class_scores) | set(REGIONS))
    fused = {}
    for r in regions:
        raw = alpha * lead_scores.get(r, 0.0) + (1 - alpha) * class_scores.get(r, 0.0)
        fused[r] = float(raw * max(0.0, min(1.0, sqi)))
    # NORM strong? keep all low if no abnormal class probability higher than 0.35 handled by caller optional
    return fused


def region_from_leads(leads: list[str]) -> list[str]:
    return sorted({LEAD_TO_REGION[l] for l in leads if l in LEAD_TO_REGION})
