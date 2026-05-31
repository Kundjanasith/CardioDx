from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

from cardiotwin.constants import PTBXL_SUPERCLASSES

out = Path("artifacts/hyp_improvement_v21")
out.mkdir(parents=True, exist_ok=True)

P = np.load("artifacts/external_validation/georgia_deep_inceptiontime_v21/P_georgia_deep.npy")
Y = np.load("artifacts/external_validation/georgia_deep_inceptiontime_v21/Y_georgia_deep.npy").astype(int)

j = PTBXL_SUPERCLASSES.index("HYP")
yt = Y[:, j]
pp = P[:, j]

rows = []
for th in np.round(np.arange(0.02, 0.96, 0.01), 2):
    yp = (pp >= th).astype(int)
    rows.append({
        "threshold": float(th),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "sensitivity": float(recall_score(yt, yp, zero_division=0)),
        "pred_positive_rate": float(yp.mean()),
    })

df = pd.DataFrame(rows)
df.to_csv(out / "hyp_threshold_analysis.csv", index=False)

best_f1 = df.sort_values("f1", ascending=False).iloc[0].to_dict()
best_sensitivity_60 = df[df["sensitivity"] >= 0.60].sort_values(["f1", "precision"], ascending=False)
best_sensitivity_70 = df[df["sensitivity"] >= 0.70].sort_values(["f1", "precision"], ascending=False)

report = {
    "label": "HYP",
    "support": int(yt.sum()),
    "current_threshold_0p5": df[df["threshold"] == 0.5].iloc[0].to_dict() if (df["threshold"] == 0.5).any() else None,
    "best_f1_threshold": best_f1,
    "best_threshold_with_sensitivity_ge_0p60": best_sensitivity_60.iloc[0].to_dict() if len(best_sensitivity_60) else None,
    "best_threshold_with_sensitivity_ge_0p70": best_sensitivity_70.iloc[0].to_dict() if len(best_sensitivity_70) else None,
    "auroc": float(roc_auc_score(yt, pp)),
    "auprc": float(average_precision_score(yt, pp)),
    "improvement_plan": [
        "Use HYP-specific threshold from deep safety profile instead of fixed 0.5.",
        "Add voltage/morphology features: LVH/RVH voltage proxies, QRS amplitude by V1/V5/V6, limb-lead axis proxies.",
        "Add chamber-enlargement clinical evidence for LAE/LAA/RVH/LVH codes.",
        "Report HYP as lower-confidence class unless calibrated confidence and SQI pass.",
        "Consider external fine-tuning or domain calibration on Georgia/CPSC for HYP."
    ],
    "claim_boundary": "HYP remains the weakest external class and should be reported with caution until external calibration and morphology-specific features improve."
}

(out / "hyp_improvement_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

html = f"""
<html><head><meta charset="utf-8"><title>HYP Improvement Report</title></head>
<body>
<h1>CardioTwin-AI HYP Improvement Report v2.1</h1>
<p><b>Support:</b> {report["support"]}</p>
<p><b>AUROC:</b> {report["auroc"]:.4f}</p>
<p><b>AUPRC:</b> {report["auprc"]:.4f}</p>
<h2>Best F1 Threshold</h2>
<pre>{json.dumps(report["best_f1_threshold"], indent=2)}</pre>
<h2>Improvement Plan</h2>
<ul>
{''.join(f'<li>{x}</li>' for x in report["improvement_plan"])}
</ul>
<p>{report["claim_boundary"]}</p>
</body></html>
"""
(out / "hyp_improvement_report.html").write_text(html, encoding="utf-8")

print(json.dumps(report, indent=2, ensure_ascii=False))
print("Saved:", out)
