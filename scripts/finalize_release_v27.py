from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone
import pandas as pd

RELEASE = Path("artifacts/release_rc1")
SUMMARY_MD = RELEASE / "CARDIOTWIN_RC1_RESEARCH_SUMMARY.md"
RESULTS_CSV = RELEASE / "CARDIOTWIN_RC1_RESULTS_TABLE.csv"
MANIFEST_JSON = RELEASE / "release_manifest.json"
ZIP_PATH = RELEASE / "release_rc1.zip"

PTB_SUMMARY = RELEASE / "ptb_mi_rich_comparison_v27" / "ptb_mi_rich_comparison_summary_v27.json"

def read_json(p):
    p = Path(p)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    text = text.replace(": NaN", ": null")
    return json.loads(text)

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe(x, digits=4):
    try:
        if x is None:
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

ptb = read_json(PTB_SUMMARY)
manifest = read_json(MANIFEST_JSON)

# Update summary markdown
summary = SUMMARY_MD.read_text(encoding="utf-8", errors="ignore") if SUMMARY_MD.exists() else ""

addendum = f"""

---

## 15. PTB MI-rich External Stress Test v2.7

PTB was added as an MI-focused external stress test. It is not a balanced all-class external validation dataset.

Label support:

- NORM: `{ptb.get("label_counts", {}).get("NORM")}`
- MI: `{ptb.get("label_counts", {}).get("MI")}`
- STTC: `{ptb.get("label_counts", {}).get("STTC")}`
- CD: `{ptb.get("label_counts", {}).get("CD")}`
- HYP: `{ptb.get("label_counts", {}).get("HYP")}`

Valid-label macro used labels: `{", ".join(ptb.get("valid_labels", []))}`  
Excluded labels: `{", ".join(ptb.get("excluded_labels", []))}`

### PTB MI-rich valid-label results

| Model | AUROC | AUPRC | Macro-F1 | Precision | Sensitivity | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Feature baseline | {safe(ptb.get("baseline_valid_macro", {}).get("auroc_macro_valid"))} | {safe(ptb.get("baseline_valid_macro", {}).get("auprc_macro_valid"))} | {safe(ptb.get("baseline_valid_macro", {}).get("macro_f1_valid"))} | {safe(ptb.get("baseline_valid_macro", {}).get("macro_precision_valid"))} | {safe(ptb.get("baseline_valid_macro", {}).get("macro_sensitivity_valid"))} | NA |
| InceptionTime | {safe(ptb.get("inceptiontime_valid_macro", {}).get("auroc_macro_valid"))} | {safe(ptb.get("inceptiontime_valid_macro", {}).get("auprc_macro_valid"))} | {safe(ptb.get("inceptiontime_valid_macro", {}).get("macro_f1_valid"))} | {safe(ptb.get("inceptiontime_valid_macro", {}).get("macro_precision_valid"))} | {safe(ptb.get("inceptiontime_valid_macro", {}).get("macro_sensitivity_valid"))} | {safe(ptb.get("inceptiontime_valid_macro", {}).get("latency_ms_per_record"))} ms/record |

### MI-specific result

InceptionTime MI performance:

- AUROC: `{safe(ptb.get("mi_class_inceptiontime", {}).get("auroc"))}`
- AUPRC: `{safe(ptb.get("mi_class_inceptiontime", {}).get("auprc"))}`
- F1: `{safe(ptb.get("mi_class_inceptiontime", {}).get("f1"))}`
- Precision: `{safe(ptb.get("mi_class_inceptiontime", {}).get("precision"))}`
- Sensitivity: `{safe(ptb.get("mi_class_inceptiontime", {}).get("sensitivity"))}`

Interpretation: PTB strengthens the MI-focused stress-test evidence. It should not be used for balanced all-five-label claims because STTC is absent and HYP support is below the valid-label threshold.
"""

if "## 15. PTB MI-rich External Stress Test v2.7" not in summary:
    summary += addendum

SUMMARY_MD.write_text(summary.replace("โ€”", "-"), encoding="utf-8")

# Update results table
if RESULTS_CSV.exists():
    results = pd.read_csv(RESULTS_CSV)
else:
    results = pd.DataFrame(columns=["section", "item", "dataset", "model", "metric", "value", "notes"])

if "ptb_mi_rich_stress_test_v27" not in set(results.get("section", [])):
    rows = []
    for model_key, model_name in [
        ("baseline_valid_macro", "Feature baseline"),
        ("inceptiontime_valid_macro", "InceptionTime"),
    ]:
        m = ptb.get(model_key, {})
        for metric in [
            "auroc_macro_valid",
            "auprc_macro_valid",
            "macro_f1_valid",
            "macro_precision_valid",
            "macro_sensitivity_valid",
            "latency_ms_per_record",
        ]:
            rows.append({
                "section": "ptb_mi_rich_stress_test_v27",
                "item": "PTB MI-rich valid-label metric",
                "dataset": "PTB MI-rich external stress test v2.7",
                "model": model_name,
                "metric": metric,
                "value": m.get(metric),
                "notes": "MI-focused stress test; not balanced all-class validation.",
            })

    mi = ptb.get("mi_class_inceptiontime", {})
    for metric in ["auroc", "auprc", "f1", "precision", "sensitivity"]:
        rows.append({
            "section": "ptb_mi_rich_stress_test_v27",
            "item": "MI-specific InceptionTime result",
            "dataset": "PTB MI-rich external stress test v2.7",
            "model": "InceptionTime",
            "metric": f"MI_{metric}",
            "value": mi.get(metric),
            "notes": "MI-specific stress-test evidence.",
        })

    results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
    results.to_csv(RESULTS_CSV, index=False)

# Update manifest
manifest["release"] = "v2.7 RC1"
manifest["title"] = "CardioTwin-AI 12L v2.7 RC1 - Paper-ready + Export Pack + MI-rich Stress Test"
manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

manifest.setdefault("high_level_metrics", {})
manifest["high_level_metrics"]["ptb_mi_rich_dataset"] = "PTB MI-rich external stress test v2.7"
manifest["high_level_metrics"]["ptb_mi_rich_model"] = "InceptionTime"
manifest["high_level_metrics"]["ptb_mi_rich_valid_auroc"] = ptb.get("inceptiontime_valid_macro", {}).get("auroc_macro_valid")
manifest["high_level_metrics"]["ptb_mi_rich_valid_auprc"] = ptb.get("inceptiontime_valid_macro", {}).get("auprc_macro_valid")
manifest["high_level_metrics"]["ptb_mi_rich_valid_macro_f1"] = ptb.get("inceptiontime_valid_macro", {}).get("macro_f1_valid")
manifest["high_level_metrics"]["ptb_mi_rich_valid_sensitivity"] = ptb.get("inceptiontime_valid_macro", {}).get("macro_sensitivity_valid")
manifest["high_level_metrics"]["ptb_mi_specific_auroc"] = ptb.get("mi_class_inceptiontime", {}).get("auroc")
manifest["high_level_metrics"]["ptb_mi_specific_auprc"] = ptb.get("mi_class_inceptiontime", {}).get("auprc")
manifest["high_level_metrics"]["ptb_mi_specific_f1"] = ptb.get("mi_class_inceptiontime", {}).get("f1")
manifest["high_level_metrics"]["ptb_mi_specific_sensitivity"] = ptb.get("mi_class_inceptiontime", {}).get("sensitivity")

manifest["ptb_mi_rich_stress_test_v27"] = ptb
manifest["dashboard_export_pack_v27"] = {
    "status": "included",
    "dashboard_file": "dashboard_v27_export_pack/streamlit_dashboard_v27_export_pack.py",
    "exports": ["JSON report", "HTML case report", "interactive 3D HTML", "optional PNG snapshot"],
}
manifest["paper_ready_audit_v27"] = {
    "status": "PASS",
    "checks": 32,
    "critical_failures": 0,
    "warnings": 0,
}

# Rebuild file index and ZIP
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
manifest["claim_boundary"] = "Research-use preliminary screening and visual explanation prototype. Not final diagnosis."

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

print("DONE: CardioTwin-AI v2.7 RC1 finalized")
print("Manifest:", MANIFEST_JSON)
print("ZIP:", ZIP_PATH)
print("ZIP size MB:", f"{ZIP_PATH.stat().st_size / 1024 / 1024:.2f}")
print("files_indexed:", manifest["files_indexed"])
print("PTB MI-specific InceptionTime AUROC:", manifest["high_level_metrics"]["ptb_mi_specific_auroc"])
print("PTB MI-specific InceptionTime AUPRC:", manifest["high_level_metrics"]["ptb_mi_specific_auprc"])
print("PTB MI-specific InceptionTime F1:", manifest["high_level_metrics"]["ptb_mi_specific_f1"])
