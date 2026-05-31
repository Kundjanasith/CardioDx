from pathlib import Path
import json
import math
import pandas as pd

OUT = Path("artifacts/external_validation/ptb_mi_rich_comparison_v27")
OUT.mkdir(parents=True, exist_ok=True)

def read_json(p):
    p = Path(p)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    # tolerate NaN in older JSON outputs
    text = text.replace(": NaN", ": null")
    return json.loads(text)

def clean_float(x):
    try:
        if x is None:
            return None
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None

base = read_json("artifacts/external_validation/ptb_mi_rich_true_eval_v27/georgia_external_metrics.json")
deep = read_json("artifacts/external_validation/ptb_mi_rich_deep_inceptiontime_v27/georgia_deep_external_metrics.json")

base_pc = pd.read_csv("artifacts/external_validation/ptb_mi_rich_true_eval_v27/georgia_metrics_per_class.csv")
deep_pc = pd.read_csv("artifacts/external_validation/ptb_mi_rich_deep_inceptiontime_v27/georgia_deep_metrics_per_class.csv")

support_threshold = 20
valid_labels = deep.get("valid_label_metrics", {}).get("valid_labels", [])
if not valid_labels:
    valid_labels = deep_pc[deep_pc["support"] >= support_threshold]["label"].tolist()

def valid_macro_from_per_class(df, labels):
    valid = df[df["label"].isin(labels)].copy()
    metrics = {}
    for col in ["auroc", "auprc", "f1", "precision", "sensitivity"]:
        vals = pd.to_numeric(valid[col], errors="coerce").dropna()
        metrics[col] = float(vals.mean()) if len(vals) else None
    return metrics

base_valid = valid_macro_from_per_class(base_pc, valid_labels)
deep_valid = deep.get("valid_label_metrics", {})

rows = [
    {
        "dataset": "PTB MI-rich external stress test v2.7",
        "model": "Feature baseline",
        "label_scope": ",".join(valid_labels),
        "auroc_macro_valid": base_valid.get("auroc"),
        "auprc_macro_valid": base_valid.get("auprc"),
        "macro_f1_valid": base_valid.get("f1"),
        "macro_precision_valid": base_valid.get("precision"),
        "macro_sensitivity_valid": base_valid.get("sensitivity"),
        "overall_auprc_macro_threshold_0p5": base.get("overall_metrics_threshold_0p5", {}).get("auprc_macro"),
        "overall_macro_f1_threshold_0p5": base.get("overall_metrics_threshold_0p5", {}).get("macro_f1"),
        "overall_macro_sensitivity_threshold_0p5": base.get("overall_metrics_threshold_0p5", {}).get("macro_recall_sensitivity"),
        "latency_ms_per_record": None,
    },
    {
        "dataset": "PTB MI-rich external stress test v2.7",
        "model": "InceptionTime",
        "label_scope": ",".join(valid_labels),
        "auroc_macro_valid": deep_valid.get("macro_auroc_valid"),
        "auprc_macro_valid": deep_valid.get("macro_auprc_valid"),
        "macro_f1_valid": deep_valid.get("macro_f1_valid"),
        "macro_precision_valid": deep_valid.get("macro_precision_valid"),
        "macro_sensitivity_valid": deep_valid.get("macro_sensitivity_valid"),
        "overall_auprc_macro_threshold_0p5": deep.get("overall_metrics_threshold_0p5", {}).get("auprc_macro"),
        "overall_macro_f1_threshold_0p5": deep.get("overall_metrics_threshold_0p5", {}).get("macro_f1"),
        "overall_macro_sensitivity_threshold_0p5": deep.get("overall_metrics_threshold_0p5", {}).get("macro_recall_sensitivity"),
        "latency_ms_per_record": deep.get("inference_latency_ms_per_record"),
    },
]

df = pd.DataFrame(rows)
df.to_csv(OUT / "ptb_mi_rich_model_comparison_v27.csv", index=False)
df.to_markdown(OUT / "ptb_mi_rich_model_comparison_v27.md", index=False)

gains = {}
for key in ["auroc_macro_valid", "auprc_macro_valid", "macro_f1_valid", "macro_precision_valid", "macro_sensitivity_valid"]:
    a = clean_float(rows[0].get(key))
    b = clean_float(rows[1].get(key))
    gains[key.replace("_valid", "_gain")] = None if a is None or b is None else b - a

mi_base = base_pc[base_pc["label"] == "MI"].iloc[0].to_dict()
mi_deep = deep_pc[deep_pc["label"] == "MI"].iloc[0].to_dict()

summary = {
    "version": "ptb_mi_rich_v27",
    "dataset": "PTB MI-rich external stress test v2.7",
    "purpose": "MI-focused external stress test; not balanced all-class validation.",
    "n_header_files_seen": deep.get("n_header_files_seen"),
    "n_usable_records": deep.get("n_usable_records"),
    "n_skipped_records": deep.get("n_skipped_records"),
    "label_counts": deep.get("label_counts"),
    "valid_support_threshold": support_threshold,
    "valid_labels": valid_labels,
    "excluded_labels": deep_valid.get("excluded_labels"),
    "baseline_valid_macro": rows[0],
    "inceptiontime_valid_macro": rows[1],
    "valid_macro_gains": gains,
    "mi_class_baseline": mi_base,
    "mi_class_inceptiontime": mi_deep,
    "interpretation": (
        "PTB is MI-rich and should be used as an MI-focused stress test. "
        "InceptionTime strongly improves MI discrimination and sensitivity over the feature baseline. "
        "STTC is absent and HYP support is below the valid-label threshold, so balanced all-class claims should not be made from PTB."
    ),
    "claim_boundary": "Research-use external stress test, not official CinC scoring and not final diagnosis."
}

(OUT / "ptb_mi_rich_comparison_summary_v27.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(df.to_string(index=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("Saved:", OUT)
