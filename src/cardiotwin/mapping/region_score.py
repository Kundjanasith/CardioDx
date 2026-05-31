from __future__ import annotations
from cardiotwin.mapping.lead_region_map import lead_to_region_scores, class_prior_region_scores, fuse_region_scores


def build_region_risk(class_probabilities: dict, lead_importance: dict, sqi: float = 1.0) -> dict:
    lead_scores = lead_to_region_scores(lead_importance)
    class_scores = class_prior_region_scores(class_probabilities)
    fused = fuse_region_scores(lead_scores, class_scores, sqi=sqi)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "region_risk": fused,
        "ranked_regions": [{"region": k, "risk": float(v)} for k, v in ranked],
        "lead_component": lead_scores,
        "class_prior_component": class_scores,
    }
