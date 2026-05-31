from __future__ import annotations
import numpy as np
import pandas as pd
from cardiotwin.constants import LEAD_TO_REGION, CLASS_TO_REGION_PRIOR, REGIONS

CLASS_EVIDENCE_REGIONS = {
    "MI": ["inferior", "anterior", "septal", "lateral"],
    "STTC": ["inferior", "anterior", "septal", "lateral"],
    "CD": ["global_conduction", "septal"],
    "HYP": ["lateral", "anterior", "septal"],
    "NORM": [],
}

def lead_region_agreement(lead_importance: dict, region_risk: dict) -> float:
    if not lead_importance:
        return 0.0
    lead_mass = {r: 0.0 for r in REGIONS}
    for lead, score in lead_importance.items():
        lead_mass[LEAD_TO_REGION.get(lead, "uncertain")] += float(score)
    regions = sorted(set(lead_mass) | set(region_risk))
    a = np.array([lead_mass.get(r, 0.0) for r in regions])
    b = np.array([region_risk.get(r, 0.0) for r in regions])
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def clinical_rule_agreement(class_probs: dict, region_risk: dict) -> dict:
    rows = []
    for label, prob in class_probs.items():
        allowed = CLASS_EVIDENCE_REGIONS.get(label, [])
        if not allowed:
            continue
        mass_allowed = sum(float(region_risk.get(r, 0.0)) for r in allowed)
        mass_total = sum(float(v) for v in region_risk.values()) + 1e-9
        rows.append({
            "label": label,
            "class_probability": float(prob),
            "allowed_regions": ",".join(allowed),
            "region_alignment": float(mass_allowed / mass_total),
        })
    overall = float(np.average([r["region_alignment"] for r in rows], weights=[max(r["class_probability"], 1e-3) for r in rows])) if rows else 0.0
    return {"overall_clinical_rule_agreement": overall, "rows": rows}

def agreement_rows_to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
