from pathlib import Path
import json
import pandas as pd
import math

ROOT = Path(".")
OUT = ROOT / "artifacts" / "release_rc1"
OUT.mkdir(parents=True, exist_ok=True)

def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def safe(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    except Exception:
        return None

rows = []

# 1) Feature baseline | PTB-XL internal
ptbxl_base = read_json("artifacts/metrics/metrics_overall.json")
rows.append({
    "model": "Feature baseline",
    "dataset": "PTB-XL internal",
    "validation_type": "internal",
    "label_scope": "all_5_labels",
    "auroc_macro": safe(ptbxl_base.get("auroc_macro")),
    "auprc_macro": safe(ptbxl_base.get("auprc_macro")),
    "macro_f1": safe(ptbxl_base.get("macro_f1")),
    "macro_precision": safe(ptbxl_base.get("macro_precision")),
    "macro_sensitivity": safe(ptbxl_base.get("macro_recall_sensitivity")),
    "latency_ms_per_record": safe(ptbxl_base.get("efficiency_latency_ms_mean")),
    "model_size_mb": safe(ptbxl_base.get("efficiency_model_size_mb")),
    "notes": "Ultra-light feature baseline; CPU-only."
})

# 2) InceptionTime | PTB-XL internal
deep_lb_path = Path("artifacts/deep_models/model_leaderboard.csv")
deep_metrics = read_json("artifacts/deep_models/deep_metrics.json")
inc_row = {}

if deep_lb_path.exists():
    lb = pd.read_csv(deep_lb_path)
    # robust model-name match
    candidates = lb[lb.astype(str).apply(lambda col: col.str.contains("inception", case=False, na=False)).any(axis=1)]
    if len(candidates) > 0:
        inc_row = candidates.iloc[0].to_dict()

def pick_metric(d, names):
    for n in names:
        if n in d:
            return safe(d[n])
    return None

# Fallback known structure / possible column names
rows.append({
    "model": "InceptionTime",
    "dataset": "PTB-XL internal",
    "validation_type": "internal",
    "label_scope": "all_5_labels",
    "auroc_macro": pick_metric(inc_row, ["auroc_macro", "auroc", "AUROC", "macro_auroc"]),
    "auprc_macro": pick_metric(inc_row, ["auprc_macro", "auprc", "AUPRC", "macro_auprc"]),
    "macro_f1": pick_metric(inc_row, ["macro_f1", "f1", "Macro-F1"]),
    "macro_precision": pick_metric(inc_row, ["macro_precision", "precision"]),
    "macro_sensitivity": pick_metric(inc_row, ["macro_recall_sensitivity", "recall", "sensitivity"]),
    "latency_ms_per_record": None,
    "model_size_mb": (Path("artifacts/deep_models/inceptiontime_model.pt").stat().st_size / 1024 / 1024) if Path("artifacts/deep_models/inceptiontime_model.pt").exists() else None,
    "notes": "Best internal deep waveform model from deep leaderboard."
})

# 3) Feature baseline | Georgia external v2.1
geo_base = read_json("artifacts/external_validation/georgia_true_eval_v21/georgia_external_metrics.json")
geo_base_valid = read_json("artifacts/external_validation/georgia_true_eval_v21/georgia_valid_label_metrics_v21.json")
rows.append({
    "model": "Feature baseline",
    "dataset": "Georgia external v2.1",
    "validation_type": "external",
    "label_scope": "valid_labels_support>=20",
    "auroc_macro": safe(geo_base_valid.get("macro_auroc_valid")),
    "auprc_macro": safe(geo_base_valid.get("macro_auprc_valid")),
    "macro_f1": safe(geo_base_valid.get("macro_f1_valid")),
    "macro_precision": safe(geo_base_valid.get("macro_precision_valid")),
    "macro_sensitivity": safe(geo_base_valid.get("macro_sensitivity_valid")),
    "latency_ms_per_record": None,
    "model_size_mb": safe(ptbxl_base.get("efficiency_model_size_mb")),
    "notes": "Georgia external evaluation using harmonization v2.1; MI excluded from valid-label macro due to low support."
})

# 4) InceptionTime | Georgia external v2.1
geo_deep = read_json("artifacts/external_validation/georgia_deep_inceptiontime_v21/georgia_deep_external_metrics.json")
geo_deep_valid = geo_deep.get("valid_label_metrics", {}) or read_json("artifacts/external_validation/georgia_deep_inceptiontime_v21/georgia_deep_valid_label_metrics.json")
rows.append({
    "model": "InceptionTime",
    "dataset": "Georgia external v2.1",
    "validation_type": "external",
    "label_scope": "valid_labels_support>=20",
    "auroc_macro": safe(geo_deep_valid.get("macro_auroc_valid")),
    "auprc_macro": safe(geo_deep_valid.get("macro_auprc_valid")),
    "macro_f1": safe(geo_deep_valid.get("macro_f1_valid")),
    "macro_precision": safe(geo_deep_valid.get("macro_precision_valid")),
    "macro_sensitivity": safe(geo_deep_valid.get("macro_sensitivity_valid")),
    "latency_ms_per_record": safe(geo_deep.get("inference_latency_ms_per_record")),
    "model_size_mb": (Path("artifacts/deep_models/inceptiontime_model.pt").stat().st_size / 1024 / 1024) if Path("artifacts/deep_models/inceptiontime_model.pt").exists() else None,
    "notes": "Deep waveform model evaluated on Georgia external v2.1 harmonization; CPU inference."
})

df = pd.DataFrame(rows)

# Add gain columns for external comparison
try:
    base_ext = df[(df["model"] == "Feature baseline") & (df["dataset"] == "Georgia external v2.1")].iloc[0]
    deep_ext = df[(df["model"] == "InceptionTime") & (df["dataset"] == "Georgia external v2.1")].iloc[0]
    gains = {
        "external_valid_auroc_gain_inceptiontime_vs_feature": safe(deep_ext["auroc_macro"] - base_ext["auroc_macro"]),
        "external_valid_auprc_gain_inceptiontime_vs_feature": safe(deep_ext["auprc_macro"] - base_ext["auprc_macro"]),
        "external_valid_macro_f1_gain_inceptiontime_vs_feature": safe(deep_ext["macro_f1"] - base_ext["macro_f1"]),
    }
except Exception:
    gains = {}

leaderboard_path = OUT / "model_dataset_comparison_leaderboard.csv"
summary_path = OUT / "model_dataset_comparison_summary.json"

df.to_csv(leaderboard_path, index=False)

summary = {
    "leaderboard_path": str(leaderboard_path),
    "n_rows": int(len(df)),
    "best_external_model_by_valid_auroc": df[df["validation_type"] == "external"].sort_values("auroc_macro", ascending=False).iloc[0].to_dict(),
    "best_external_model_by_valid_auprc": df[df["validation_type"] == "external"].sort_values("auprc_macro", ascending=False).iloc[0].to_dict(),
    "best_external_model_by_valid_macro_f1": df[df["validation_type"] == "external"].sort_values("macro_f1", ascending=False).iloc[0].to_dict(),
    "gains": gains,
    "claim_boundary": "Georgia external metrics use CinC 2020 Georgia subset with v2.1 harmonization; valid-label macro excludes labels with support < 20.",
}

summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(df.to_string(index=False))
print("Saved:", leaderboard_path)
print("Saved:", summary_path)
print(json.dumps(summary, indent=2, ensure_ascii=False))
