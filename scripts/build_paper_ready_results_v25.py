from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

out = Path("artifacts/paper_ready_v25")
out.mkdir(parents=True, exist_ok=True)

def read_json(p):
    p = Path(p)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

leaderboard_path = Path("artifacts/release_rc1/model_dataset_comparison_leaderboard.csv")
if not leaderboard_path.exists():
    raise FileNotFoundError(f"Missing {leaderboard_path}")

leaderboard = pd.read_csv(leaderboard_path)

cpsc_base = read_json("artifacts/release_rc1/cpsc2018_external_baseline_v21/cpsc2018_valid_label_metrics_baseline_v21.json")
cpsc_deep = read_json("artifacts/release_rc1/cpsc2018_external_inceptiontime_v21/georgia_deep_external_metrics.json")
cpsc_deep_valid = cpsc_deep.get("valid_label_metrics", {})

rows = []

for _, r in leaderboard.iterrows():
    rows.append({
        "model": r.get("model"),
        "dataset": r.get("dataset"),
        "validation_type": r.get("validation_type"),
        "label_scope": r.get("label_scope"),
        "auroc_macro": r.get("auroc_macro"),
        "auprc_macro": r.get("auprc_macro"),
        "macro_f1": r.get("macro_f1"),
        "macro_sensitivity": r.get("macro_sensitivity"),
        "latency_ms_per_record": r.get("latency_ms_per_record"),
        "notes": r.get("notes"),
    })

rows.append({
    "model": "Feature baseline",
    "dataset": "CPSC 2018 external v2.1",
    "validation_type": "external",
    "label_scope": "valid_labels_NORM_STTC_CD",
    "auroc_macro": cpsc_base.get("macro_auroc_valid"),
    "auprc_macro": cpsc_base.get("macro_auprc_valid"),
    "macro_f1": cpsc_base.get("macro_f1_valid"),
    "macro_sensitivity": cpsc_base.get("macro_sensitivity_valid"),
    "latency_ms_per_record": None,
    "notes": "Valid labels: NORM/STTC/CD; MI/HYP excluded due to zero mapped support.",
})

rows.append({
    "model": "InceptionTime",
    "dataset": "CPSC 2018 external v2.1",
    "validation_type": "external",
    "label_scope": "valid_labels_NORM_STTC_CD",
    "auroc_macro": cpsc_deep_valid.get("macro_auroc_valid"),
    "auprc_macro": cpsc_deep_valid.get("macro_auprc_valid"),
    "macro_f1": cpsc_deep_valid.get("macro_f1_valid"),
    "macro_sensitivity": cpsc_deep_valid.get("macro_sensitivity_valid"),
    "latency_ms_per_record": cpsc_deep.get("inference_latency_ms_per_record"),
    "notes": "Valid labels: NORM/STTC/CD; CPU inference.",
})

df = pd.DataFrame(rows)
df.to_csv(out / "table1_model_comparison_with_cpsc2018.csv", index=False)
df.to_markdown(out / "table1_model_comparison_with_cpsc2018.md", index=False)

plot_df = df.copy()
plot_df["name"] = plot_df["model"].astype(str) + "\n" + plot_df["dataset"].astype(str)

for metric in ["auroc_macro", "auprc_macro", "macro_f1", "macro_sensitivity"]:
    tmp = plot_df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")

    plt.figure(figsize=(13, 6))
    plt.bar(tmp["name"], tmp[metric])
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.title(f"CardioTwin-AI v2.5 Model Comparison: {metric}")
    plt.tight_layout()
    plt.savefig(out / f"fig_v25_model_comparison_{metric}.png", dpi=300)
    plt.close()

cpsc_pc_path = Path("artifacts/release_rc1/cpsc2018_external_inceptiontime_v21/georgia_deep_metrics_per_class.csv")
if cpsc_pc_path.exists():
    cpsc_deep_pc = pd.read_csv(cpsc_pc_path)
    cpsc_deep_pc.to_csv(out / "table2_cpsc2018_inceptiontime_per_class.csv", index=False)
    cpsc_deep_pc.to_markdown(out / "table2_cpsc2018_inceptiontime_per_class.md", index=False)

    for metric in ["auroc", "auprc", "f1", "sensitivity"]:
        tmp = cpsc_deep_pc.copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce").fillna(0)

        plt.figure(figsize=(8, 5))
        plt.bar(tmp["label"], tmp[metric])
        plt.ylim(0, 1)
        plt.ylabel(metric)
        plt.title(f"CPSC 2018 InceptionTime Per-class {metric}")
        plt.tight_layout()
        plt.savefig(out / f"fig_cpsc2018_inceptiontime_per_class_{metric}.png", dpi=300)
        plt.close()

summary = {
    "version": "paper_ready_v25",
    "outputs": [
        "table1_model_comparison_with_cpsc2018.csv",
        "table1_model_comparison_with_cpsc2018.md",
        "fig_v25_model_comparison_auroc_macro.png",
        "fig_v25_model_comparison_auprc_macro.png",
        "fig_v25_model_comparison_macro_f1.png",
        "fig_v25_model_comparison_macro_sensitivity.png",
        "table2_cpsc2018_inceptiontime_per_class.csv",
        "table2_cpsc2018_inceptiontime_per_class.md",
        "fig_cpsc2018_inceptiontime_per_class_auroc.png",
        "fig_cpsc2018_inceptiontime_per_class_auprc.png",
        "fig_cpsc2018_inceptiontime_per_class_f1.png",
        "fig_cpsc2018_inceptiontime_per_class_sensitivity.png",
    ],
    "caption_notes": {
        "model_comparison": "InceptionTime outperforms the feature baseline on both Georgia and CPSC 2018 valid-label external validation.",
        "cpsc_per_class": "CPSC 2018 valid labels under harmonization v2.1 are NORM, STTC, and CD; MI and HYP have zero mapped positive support."
    }
}

(out / "paper_ready_v25_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Saved paper-ready v2.5 outputs:", out)
print(json.dumps(summary, indent=2, ensure_ascii=False))
