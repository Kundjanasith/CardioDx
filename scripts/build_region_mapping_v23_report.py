from pathlib import Path
import json
import pandas as pd

from cardiotwin.explain.region_mapper_v23 import map_prediction_to_region

out = Path("artifacts/region_mapping_v23")
out.mkdir(parents=True, exist_ok=True)

# Synthetic sanity cases to verify that lateral does not dominate simply because it has many leads.
cases = [
    {
        "case_id": "clear_lateral",
        "predicted_class": "STTC",
        "prob": 0.88,
        "lead_scores": {"I": 0.8, "aVL": 0.7, "V5": 0.9, "V6": 0.85},
    },
    {
        "case_id": "clear_inferior",
        "predicted_class": "STTC",
        "prob": 0.88,
        "lead_scores": {"II": 0.9, "III": 0.85, "aVF": 0.95, "I": 0.1, "aVL": 0.1},
    },
    {
        "case_id": "clear_anterior",
        "predicted_class": "MI",
        "prob": 0.82,
        "lead_scores": {"V2": 0.8, "V3": 0.9, "V4": 0.7, "V5": 0.15},
    },
    {
        "case_id": "conduction_global",
        "predicted_class": "CD",
        "prob": 0.91,
        "lead_scores": {"I": 0.4, "II": 0.5, "V1": 0.55, "V2": 0.5, "V5": 0.45, "V6": 0.4},
    },
    {
        "case_id": "ambiguous_lateral_inferior",
        "predicted_class": "STTC",
        "prob": 0.75,
        "lead_scores": {"II": 0.55, "III": 0.55, "aVF": 0.55, "I": 0.55, "aVL": 0.55, "V5": 0.55, "V6": 0.55},
    },
]

rows = []
for c in cases:
    d = map_prediction_to_region(c["lead_scores"], c["predicted_class"], c["prob"])
    rows.append({
        "case_id": c["case_id"],
        "predicted_class": c["predicted_class"],
        "prob": c["prob"],
        "region": d["region"],
        "reason": d["reason"],
        "confidence": d["confidence"],
        "top_region": d.get("top_region", ""),
        "second_region": d.get("second_region", ""),
        "margin": d.get("margin", ""),
        "region_scores_json": json.dumps(d["region_scores"], ensure_ascii=False),
    })

df = pd.DataFrame(rows)
df.to_csv(out / "region_mapper_v23_sanity_cases.csv", index=False)

summary = {
    "version": "region_mapper_v23",
    "fixes": [
        "normalize region evidence by number of leads",
        "margin-based uncertain region decision",
        "class-aware priors for CD/HYP/STTC/MI",
        "reduced lateral dominance from lead-count advantage"
    ],
    "outputs": [
        "region_mapper_v23_sanity_cases.csv",
        "region_mapper_v23_summary.json"
    ],
    "recommendation": "Use region_mapper_v23 for dashboard and 3D/4D heatmap region assignment."
}

(out / "region_mapper_v23_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(df.to_string(index=False))
print("Saved:", out)
