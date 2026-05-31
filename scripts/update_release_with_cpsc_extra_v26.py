from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt


RELEASE = Path("artifacts/release_rc1")
PAPER = Path("artifacts/paper_ready_v26")
PAPER.mkdir(parents=True, exist_ok=True)

SUMMARY_MD = RELEASE / "CARDIOTWIN_RC1_RESEARCH_SUMMARY.md"
RESULTS_CSV = RELEASE / "CARDIOTWIN_RC1_RESULTS_TABLE.csv"
MANIFEST_JSON = RELEASE / "release_manifest.json"
ZIP_PATH = RELEASE / "release_rc1.zip"

EXTRA_COMP_JSON = RELEASE / "cpsc2018_extra_comparison_v25" / "cpsc2018_extra_comparison_summary_v25.json"
EXTRA_COMP_CSV = RELEASE / "cpsc2018_extra_comparison_v25" / "cpsc2018_extra_model_comparison_v25.csv"
EXTRA_DEEP_PC = RELEASE / "cpsc2018_extra_external_inceptiontime_v25" / "georgia_deep_metrics_per_class.csv"


def read_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def safe(x, digits=4):
    try:
        if x is None:
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


extra = read_json(EXTRA_COMP_JSON)
baseline = extra.get("baseline", {})
deep = extra.get("inceptiontime", {})
gains = extra.get("gains", {})
label_support = extra.get("label_support", {})

# ---------------------------------------------------------------------
# 1) Update results table
# ---------------------------------------------------------------------
if RESULTS_CSV.exists():
    results = pd.read_csv(RESULTS_CSV)
else:
    results = pd.DataFrame(columns=["section", "item", "dataset", "model", "metric", "value", "notes"])

new_rows = [
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Label support",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "",
        "metric": "label_counts",
        "value": json.dumps(label_support, ensure_ascii=False),
        "notes": "First external subset with all five target labels evaluable under harmonization v2.1.",
    },
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Valid AUROC",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "Feature baseline",
        "metric": "auroc_macro",
        "value": baseline.get("auroc_macro"),
        "notes": "All five labels evaluable.",
    },
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Valid AUPRC",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "Feature baseline",
        "metric": "auprc_macro",
        "value": baseline.get("auprc_macro"),
        "notes": "All five labels evaluable.",
    },
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Valid Macro-F1",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "Feature baseline",
        "metric": "macro_f1",
        "value": baseline.get("macro_f1"),
        "notes": "All five labels evaluable.",
    },
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Valid sensitivity",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "Feature baseline",
        "metric": "macro_sensitivity",
        "value": baseline.get("macro_sensitivity"),
        "notes": "All five labels evaluable.",
    },
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Valid AUROC",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "InceptionTime",
        "metric": "auroc_macro",
        "value": deep.get("auroc_macro"),
        "notes": "All five labels evaluable; CPU inference.",
    },
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Valid AUPRC",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "InceptionTime",
        "metric": "auprc_macro",
        "value": deep.get("auprc_macro"),
        "notes": "All five labels evaluable; CPU inference.",
    },
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Valid Macro-F1",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "InceptionTime",
        "metric": "macro_f1",
        "value": deep.get("macro_f1"),
        "notes": "All five labels evaluable; CPU inference.",
    },
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Valid sensitivity",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "InceptionTime",
        "metric": "macro_sensitivity",
        "value": deep.get("macro_sensitivity"),
        "notes": "All five labels evaluable; CPU inference.",
    },
    {
        "section": "cpsc2018_extra_external_validation",
        "item": "Latency",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "InceptionTime",
        "metric": "latency_ms_per_record",
        "value": deep.get("latency_ms_per_record"),
        "notes": "CPU ms/record.",
    },
    {
        "section": "cpsc2018_extra_gain",
        "item": "AUROC gain",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "InceptionTime vs Feature baseline",
        "metric": "auroc_gain",
        "value": gains.get("auroc_gain"),
        "notes": "Deep waveform gain over feature baseline.",
    },
    {
        "section": "cpsc2018_extra_gain",
        "item": "AUPRC gain",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "InceptionTime vs Feature baseline",
        "metric": "auprc_gain",
        "value": gains.get("auprc_gain"),
        "notes": "Deep waveform gain over feature baseline.",
    },
    {
        "section": "cpsc2018_extra_gain",
        "item": "Macro-F1 gain",
        "dataset": "CPSC 2018 Extra external v2.5",
        "model": "InceptionTime vs Feature baseline",
        "metric": "macro_f1_gain",
        "value": gains.get("macro_f1_gain"),
        "notes": "Deep waveform gain over feature baseline.",
    },
]

# avoid duplicate append if rerun
if "cpsc2018_extra_external_validation" not in set(results.get("section", [])):
    results = pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)

results.to_csv(RESULTS_CSV, index=False)

# ---------------------------------------------------------------------
# 2) Create paper-ready v2.6 table + figures
# ---------------------------------------------------------------------
comp_rows = []

old_leaderboard = RELEASE / "model_dataset_comparison_leaderboard.csv"
if old_leaderboard.exists():
    old_df = pd.read_csv(old_leaderboard)
    for _, r in old_df.iterrows():
        comp_rows.append({
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

# Add previous CPSC 2018 rows if available
cpsc2018 = RELEASE / "cpsc2018_external_inceptiontime_v21" / "georgia_deep_external_metrics.json"
cpsc2018_base = RELEASE / "cpsc2018_external_baseline_v21" / "cpsc2018_valid_label_metrics_baseline_v21.json"

if cpsc2018_base.exists():
    b = read_json(cpsc2018_base)
    comp_rows.append({
        "model": "Feature baseline",
        "dataset": "CPSC 2018 external v2.1",
        "validation_type": "external",
        "label_scope": "valid_labels_NORM_STTC_CD",
        "auroc_macro": b.get("macro_auroc_valid"),
        "auprc_macro": b.get("macro_auprc_valid"),
        "macro_f1": b.get("macro_f1_valid"),
        "macro_sensitivity": b.get("macro_sensitivity_valid"),
        "latency_ms_per_record": None,
        "notes": "MI/HYP excluded due to zero mapped support.",
    })

if cpsc2018.exists():
    d = read_json(cpsc2018)
    dv = d.get("valid_label_metrics", {})
    comp_rows.append({
        "model": "InceptionTime",
        "dataset": "CPSC 2018 external v2.1",
        "validation_type": "external",
        "label_scope": "valid_labels_NORM_STTC_CD",
        "auroc_macro": dv.get("macro_auroc_valid"),
        "auprc_macro": dv.get("macro_auprc_valid"),
        "macro_f1": dv.get("macro_f1_valid"),
        "macro_sensitivity": dv.get("macro_sensitivity_valid"),
        "latency_ms_per_record": d.get("inference_latency_ms_per_record"),
        "notes": "MI/HYP excluded due to zero mapped support; CPU inference.",
    })

# Add CPSC Extra rows
comp_rows.append({
    "model": "Feature baseline",
    "dataset": "CPSC 2018 Extra external v2.5",
    "validation_type": "external",
    "label_scope": "all_5_labels_external",
    "auroc_macro": baseline.get("auroc_macro"),
    "auprc_macro": baseline.get("auprc_macro"),
    "macro_f1": baseline.get("macro_f1"),
    "macro_sensitivity": baseline.get("macro_sensitivity"),
    "latency_ms_per_record": None,
    "notes": "First all-five-label external subset.",
})
comp_rows.append({
    "model": "InceptionTime",
    "dataset": "CPSC 2018 Extra external v2.5",
    "validation_type": "external",
    "label_scope": "all_5_labels_external",
    "auroc_macro": deep.get("auroc_macro"),
    "auprc_macro": deep.get("auprc_macro"),
    "macro_f1": deep.get("macro_f1"),
    "macro_sensitivity": deep.get("macro_sensitivity"),
    "latency_ms_per_record": deep.get("latency_ms_per_record"),
    "notes": "First all-five-label external subset; CPU inference.",
})

comp = pd.DataFrame(comp_rows)
comp.to_csv(PAPER / "table1_model_comparison_with_cpsc_extra_v26.csv", index=False)
comp.to_markdown(PAPER / "table1_model_comparison_with_cpsc_extra_v26.md", index=False)

plot_df = comp.copy()
plot_df["name"] = plot_df["model"].astype(str) + "\n" + plot_df["dataset"].astype(str)

for metric in ["auroc_macro", "auprc_macro", "macro_f1", "macro_sensitivity"]:
    tmp = plot_df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    plt.figure(figsize=(15, 6))
    plt.bar(tmp["name"], tmp[metric])
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.title(f"CardioTwin-AI v2.6 Model Comparison: {metric}")
    plt.tight_layout()
    plt.savefig(PAPER / f"fig_v26_model_comparison_{metric}.png", dpi=300)
    plt.close()

if EXTRA_DEEP_PC.exists():
    pc = pd.read_csv(EXTRA_DEEP_PC)
    pc.to_csv(PAPER / "table2_cpsc2018_extra_inceptiontime_per_class_v26.csv", index=False)
    pc.to_markdown(PAPER / "table2_cpsc2018_extra_inceptiontime_per_class_v26.md", index=False)

    for metric in ["auroc", "auprc", "f1", "sensitivity"]:
        tmp = pc.copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce").fillna(0)
        plt.figure(figsize=(8, 5))
        plt.bar(tmp["label"], tmp[metric])
        plt.ylim(0, 1)
        plt.ylabel(metric)
        plt.title(f"CPSC 2018 Extra InceptionTime Per-class {metric}")
        plt.tight_layout()
        plt.savefig(PAPER / f"fig_cpsc2018_extra_inceptiontime_per_class_{metric}.png", dpi=300)
        plt.close()

paper_summary = {
    "version": "paper_ready_v26",
    "main_update": "Added CPSC 2018 Extra all-five-label external validation.",
    "cpsc2018_extra": extra,
    "outputs": [p.name for p in sorted(PAPER.glob("*"))],
}
(PAPER / "paper_ready_v26_summary.json").write_text(
    json.dumps(paper_summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

# ---------------------------------------------------------------------
# 3) Update methods/results draft
# ---------------------------------------------------------------------
draft_v25 = RELEASE / "paper_ready_v25" / "CARDIOTWIN_METHODS_RESULTS_DRAFT_v25.md"
draft_v26 = PAPER / "CARDIOTWIN_METHODS_RESULTS_DRAFT_v26.md"

if draft_v25.exists():
    text = draft_v25.read_text(encoding="utf-8")
else:
    text = "# CardioTwin-AI 12L Methods and Results Draft\n"

addendum = f"""

---

## v2.6 Addendum: CPSC 2018 Extra All-Five-Label External Validation

CPSC 2018 Extra was added as an additional external validation subset because header probing showed stronger MI and HYP support than standard CPSC 2018. Under harmonization v2.1, this subset is the first external dataset in the current project where all five target labels are evaluable.

### Label support

- NORM: `{label_support.get("NORM")}`
- MI: `{label_support.get("MI")}`
- STTC: `{label_support.get("STTC")}`
- CD: `{label_support.get("CD")}`
- HYP: `{label_support.get("HYP")}`

### CPSC 2018 Extra external results

| Model | Labels | AUROC | AUPRC | Macro-F1 | Precision | Sensitivity | Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Feature baseline | all 5 | {safe(baseline.get("auroc_macro"))} | {safe(baseline.get("auprc_macro"))} | {safe(baseline.get("macro_f1"))} | {safe(baseline.get("macro_precision"))} | {safe(baseline.get("macro_sensitivity"))} | NA |
| InceptionTime | all 5 | {safe(deep.get("auroc_macro"))} | {safe(deep.get("auprc_macro"))} | {safe(deep.get("macro_f1"))} | {safe(deep.get("macro_precision"))} | {safe(deep.get("macro_sensitivity"))} | {safe(deep.get("latency_ms_per_record"))} ms/record |

### Gain over feature baseline

- AUROC gain: `{safe(gains.get("auroc_gain"))}`
- AUPRC gain: `{safe(gains.get("auprc_gain"))}`
- Macro-F1 gain: `{safe(gains.get("macro_f1_gain"))}`
- Sensitivity gain: `{safe(gains.get("sensitivity_gain"))}`

### Interpretation

CPSC 2018 Extra is the strongest external evidence currently available for all-five-label evaluation. Unlike standard CPSC 2018, it contains sufficient mapped positive support for MI and HYP. InceptionTime substantially outperforms the feature baseline, indicating that waveform-based deep learning is more robust than the lightweight feature baseline under this stronger cross-dataset stress test.

The result should still be reported as research-stage external validation because diagnosis harmonization remains a proxy and prospective clinical validation has not yet been performed.
"""

if "## v2.6 Addendum: CPSC 2018 Extra All-Five-Label External Validation" not in text:
    text = text + addendum

draft_v26.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------
# 4) Update release summary
# ---------------------------------------------------------------------
summary_text = SUMMARY_MD.read_text(encoding="utf-8") if SUMMARY_MD.exists() else ""

release_addendum = f"""

---

## 14. CPSC 2018 Extra All-Five-Label External Validation Addendum

CPSC 2018 Extra was added as the first external subset in this project where all five target labels are evaluable under harmonization v2.1.

Label support:

- NORM: `{label_support.get("NORM")}`
- MI: `{label_support.get("MI")}`
- STTC: `{label_support.get("STTC")}`
- CD: `{label_support.get("CD")}`
- HYP: `{label_support.get("HYP")}`

| Model | Labels | AUROC | AUPRC | Macro-F1 | Precision | Sensitivity | Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Feature baseline | all 5 | {safe(baseline.get("auroc_macro"))} | {safe(baseline.get("auprc_macro"))} | {safe(baseline.get("macro_f1"))} | {safe(baseline.get("macro_precision"))} | {safe(baseline.get("macro_sensitivity"))} | NA |
| InceptionTime | all 5 | {safe(deep.get("auroc_macro"))} | {safe(deep.get("auprc_macro"))} | {safe(deep.get("macro_f1"))} | {safe(deep.get("macro_precision"))} | {safe(deep.get("macro_sensitivity"))} | {safe(deep.get("latency_ms_per_record"))} ms/record |

InceptionTime gains over feature baseline:

- AUROC: `{safe(gains.get("auroc_gain"))}`
- AUPRC: `{safe(gains.get("auprc_gain"))}`
- Macro-F1: `{safe(gains.get("macro_f1_gain"))}`
- Sensitivity: `{safe(gains.get("sensitivity_gain"))}`

Interpretation: CPSC 2018 Extra strengthens the research claim because it evaluates all five target labels externally, including MI and HYP.
"""

if "## 14. CPSC 2018 Extra All-Five-Label External Validation Addendum" not in summary_text:
    summary_text += release_addendum
SUMMARY_MD.write_text(summary_text.replace("โ€”", "-"), encoding="utf-8")

# Copy paper-ready v26 into release
release_paper_v26 = RELEASE / "paper_ready_v26"
release_paper_v26.mkdir(parents=True, exist_ok=True)
for p in PAPER.glob("*"):
    if p.is_file():
        (release_paper_v26 / p.name).write_bytes(p.read_bytes())

# ---------------------------------------------------------------------
# 5) Update manifest + ZIP
# ---------------------------------------------------------------------
manifest = read_json(MANIFEST_JSON)

manifest["release"] = "v2.6 RC1"
manifest["title"] = "CardioTwin-AI 12L v2.6 RC1 - External-Validated + Safety-Calibrated Deep ECG Screening Prototype"
manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

manifest.setdefault("high_level_metrics", {})
manifest["high_level_metrics"]["best_external_model"] = "InceptionTime"
manifest["high_level_metrics"]["best_external_dataset"] = "CPSC 2018 external v2.1"
manifest["high_level_metrics"]["best_external_valid_auroc"] = 0.8535993614864067
manifest["high_level_metrics"]["best_all_five_external_dataset"] = "CPSC 2018 Extra external v2.5"
manifest["high_level_metrics"]["best_all_five_external_model"] = "InceptionTime"
manifest["high_level_metrics"]["best_all_five_external_valid_auroc"] = deep.get("auroc_macro")
manifest["high_level_metrics"]["best_all_five_external_valid_auprc"] = deep.get("auprc_macro")
manifest["high_level_metrics"]["best_all_five_external_valid_macro_f1"] = deep.get("macro_f1")
manifest["high_level_metrics"]["best_all_five_external_valid_sensitivity"] = deep.get("macro_sensitivity")
manifest["high_level_metrics"]["note"] = (
    "CPSC 2018 has highest external AUROC over NORM/STTC/CD. "
    "CPSC 2018 Extra is the strongest all-five-label external validation because MI and HYP are evaluable."
)

manifest["cpsc2018_extra_external_validation"] = extra
manifest["claim_boundary"] = "Research-use preliminary screening and visual explanation prototype. Not final diagnosis."

# rebuild file index excluding zip/manifest self-reference
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

files = []
for p in sorted(RELEASE.rglob("*")):
    if not p.is_file():
        continue
    if p.name in {"release_rc1.zip", "release_manifest.json"}:
        continue
    if p.name.startswith("~$"):
        continue
    st = p.stat()
    files.append({
        "path": p.relative_to(RELEASE).as_posix(),
        "size_bytes": int(st.st_size),
        "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(p),
    })

manifest["files_indexed"] = len(files)
manifest["files"] = files
manifest["manifest_note"] = "release_manifest.json and release_rc1.zip are excluded from file self-indexing to avoid stale self-referential hashes."

MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(RELEASE.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == ZIP_PATH.resolve():
            continue
        if p.name.startswith("~$"):
            continue
        z.write(p, (Path("release_rc1") / p.relative_to(RELEASE)).as_posix())

print("DONE: v2.6 release updated with CPSC 2018 Extra all-five-label external validation")
print("Summary:", SUMMARY_MD)
print("Results:", RESULTS_CSV)
print("Manifest:", MANIFEST_JSON)
print("ZIP:", ZIP_PATH)
print("ZIP size MB:", f"{ZIP_PATH.stat().st_size / 1024 / 1024:.2f}")
print("Paper-ready:", PAPER)
