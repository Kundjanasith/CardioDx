from __future__ import annotations
from datetime import datetime
from cardiotwin.mapping.region_score import build_region_risk


def build_digital_twin_state(record_id: str, class_probabilities: dict, lead_importance: dict, sqi_report: dict,
                             labels: dict | None = None) -> dict:
    sqi = float(sqi_report.get("overall_sqi", 1.0))
    region = build_region_risk(class_probabilities, lead_importance, sqi=sqi)
    top_region = region["ranked_regions"][0]["region"] if region["ranked_regions"] else "uncertain"
    risk_level = "low"
    top_risk = region["ranked_regions"][0]["risk"] if region["ranked_regions"] else 0.0
    if top_risk >= 0.65:
        risk_level = "high"
    elif top_risk >= 0.35:
        risk_level = "moderate"
    return {
        "record_id": record_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "class_probabilities": class_probabilities,
        "lead_importance": lead_importance,
        "sqi": sqi_report,
        "regions": region,
        "summary": {
            "top_region": top_region,
            "top_region_risk": float(top_risk),
            "risk_level": risk_level,
            "label_context": labels or {},
        },
        "animation": {
            "mode": "4D_explanatory_P_QRS_T",
            "phases": [
                {"phase": "P", "time_fraction": [0.00, 0.20], "highlight": ["global_rhythm"]},
                {"phase": "QRS", "time_fraction": [0.20, 0.42], "highlight": ["global_conduction", top_region]},
                {"phase": "ST", "time_fraction": [0.42, 0.62], "highlight": [top_region]},
                {"phase": "T", "time_fraction": [0.62, 1.00], "highlight": [top_region]},
            ]
        },
        "clinical_boundary": "Research-use preliminary screening and visual explanation only; not final diagnosis.",
    }
