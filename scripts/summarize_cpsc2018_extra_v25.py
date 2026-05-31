from pathlib import Path
import json
import pandas as pd

out = Path("artifacts/external_validation/cpsc2018_extra_comparison_v25")
out.mkdir(parents=True, exist_ok=True)

def read_json(p):
    p = Path(p)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

base = read_json("artifacts/external_validation/cpsc2018_extra_true_eval_v25/georgia_external_metrics.json")
deep = read_json("artifacts/external_validation/cpsc2018_extra_deep_inceptiontime_v25/georgia_deep_external_metrics.json")

base_m = base.get("overall_metrics_threshold_0p5", {})
deep_m = deep.get("valid_label_metrics", {})

rows = [
    {
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "Feature baseline",
        "valid_labels": "NORM,MI,STTC,CD,HYP",
        "auroc_macro": base_m.get("auroc_macro"),
        "auprc_macro": base_m.get("auprc_macro"),
        "macro_f1": base_m.get("macro_f1"),
        "macro_precision": base_m.get("macro_precision"),
        "macro_sensitivity": base_m.get("macro_recall_sensitivity"),
        "latency_ms_per_record": None,
    },
    {
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "InceptionTime",
        "valid_labels": ",".join(deep_m.get("valid_labels", [])),
        "auroc_macro": deep_m.get("macro_auroc_valid"),
        "auprc_macro": deep_m.get("macro_auprc_valid"),
        "macro_f1": deep_m.get("macro_f1_valid"),
        "macro_precision": deep_m.get("macro_precision_valid"),
        "macro_sensitivity": deep_m.get("macro_sensitivity_valid"),
        "latency_ms_per_record": deep.get("inference_latency_ms_per_record"),
    }
]

df = pd.DataFrame(rows)
df.to_csv(out / "cpsc2018_extra_model_comparison_v25.csv", index=False)

gain = {
    "dataset": "CPSC 2018 Extra external v2.5",
    "label_support": deep.get("label_counts", {}),
    "baseline": rows[0],
    "inceptiontime": rows[1],
    "gains": {
        "auroc_gain": rows[1]["auroc_macro"] - rows[0]["auroc_macro"],
        "auprc_gain": rows[1]["auprc_macro"] - rows[0]["auprc_macro"],
        "macro_f1_gain": rows[1]["macro_f1"] - rows[0]["macro_f1"],
        "sensitivity_gain": rows[1]["macro_sensitivity"] - rows[0]["macro_sensitivity"],
    },
    "interpretation": "CPSC 2018 Extra is the first external subset in this project with all five target labels evaluable under harmonization v2.1."
}

(out / "cpsc2018_extra_comparison_summary_v25.json").write_text(
    json.dumps(gain, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(df.to_string(index=False))
print(json.dumps(gain, indent=2, ensure_ascii=False))
print("Saved:", out)
