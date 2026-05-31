from __future__ import annotations
from cardiotwin.constants import REGIONS, LEAD_TO_REGION, CLASS_TO_REGION_PRIOR

def validated_region_risk(class_probs: dict, lead_importance: dict, temporal_evidence: dict | None = None, sqi: dict | None = None) -> dict:
    risk = {r: 0.0 for r in REGIONS}
    # Class prior contribution
    for label, prob in class_probs.items():
        for region, weight in CLASS_TO_REGION_PRIOR.get(label, {}).items():
            risk[region] += 0.30 * float(prob) * float(weight)
    # Lead evidence contribution
    for lead, imp in lead_importance.items():
        region = LEAD_TO_REGION.get(lead, "uncertain")
        risk[region] += 0.35 * float(imp)
    # Temporal proxy contribution
    if temporal_evidence:
        st_mass = temporal_evidence.get("ST_proxy", {})
        t_mass = temporal_evidence.get("T_proxy", {})
        sttc = float(st_mass.get("STTC", 0.0)) + float(t_mass.get("STTC", 0.0))
        for region in ["inferior", "anterior", "septal", "lateral"]:
            risk[region] += 0.15 * sttc
        qrs = temporal_evidence.get("QRS_proxy", {})
        risk["global_conduction"] += 0.15 * float(qrs.get("CD", 0.0))
    # SQI confidence attenuation and uncertainty routing
    sqi_value = float((sqi or {}).get("overall_sqi", 1.0))
    if sqi_value < 0.55:
        for r in risk:
            risk[r] *= 0.25
        risk["uncertain"] = max(risk.get("uncertain", 0.0), 0.9)
    elif sqi_value < 0.70:
        risk["uncertain"] = max(risk.get("uncertain", 0.0), 0.35)
    maxv = max(max(risk.values()), 1e-9)
    if maxv > 1.0:
        risk = {k: min(1.0, v / maxv) for k, v in risk.items()}
    return {k: float(v) for k, v in risk.items()}

def region_confusion_from_cases(true_regions: list[str], pred_regions: list[str]) -> dict:
    labels = sorted(set(true_regions) | set(pred_regions))
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for t, p in zip(true_regions, pred_regions):
        matrix[t][p] += 1
    return {"labels": labels, "matrix": matrix}
