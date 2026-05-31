from pathlib import Path
import json
import pandas as pd

out = Path("artifacts/external_validation/cpsc2018_true_eval_v21")
df = pd.read_csv(out / "georgia_metrics_per_class.csv")

valid = df[df["support"] >= 20].copy()

summary = {
    "dataset": "PhysioNet_CinC_2020_CPSC2018",
    "subset": "training/cpsc_2018",
    "valid_support_threshold": 20,
    "valid_labels": valid["label"].tolist(),
    "excluded_labels": df[df["support"] < 20]["label"].tolist(),
    "macro_auroc_valid": float(valid["auroc"].dropna().mean()),
    "macro_auprc_valid": float(valid["auprc"].dropna().mean()),
    "macro_f1_valid": float(valid["f1"].mean()),
    "macro_precision_valid": float(valid["precision"].mean()),
    "macro_sensitivity_valid": float(valid["sensitivity"].mean()),
    "interpretation": "Valid-label macro excludes MI and HYP because they have insufficient mapped positive support in CPSC 2018 under harmonization v2.1."
}

(out / "cpsc2018_valid_label_metrics_baseline_v21.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps(summary, indent=2, ensure_ascii=False))
