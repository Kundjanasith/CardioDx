from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone
import pandas as pd

release = Path("artifacts/release_rc1")
release.mkdir(parents=True, exist_ok=True)

summary_md = release / "CARDIOTWIN_RC1_RESEARCH_SUMMARY.md"
results_csv = release / "CARDIOTWIN_RC1_RESULTS_TABLE.csv"
manifest_json = release / "release_manifest.json"
zip_path = release / "release_rc1.zip"

def read_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def clean(x, digits=4):
    try:
        if x is None:
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

baseline = read_json("artifacts/release_rc1/cpsc2018_external_baseline_v21/cpsc2018_valid_label_metrics_baseline_v21.json")
deep = read_json("artifacts/release_rc1/cpsc2018_external_inceptiontime_v21/georgia_deep_external_metrics.json")
deep_valid = deep.get("valid_label_metrics", {})

# Append results table rows
rows = []
if results_csv.exists():
    df = pd.read_csv(results_csv)
else:
    df = pd.DataFrame(columns=["section","item","dataset","model","metric","value","notes"])

def add(section, item, dataset, model, metric, value, notes):
    rows.append({
        "section": section,
        "item": item,
        "dataset": dataset,
        "model": model,
        "metric": metric,
        "value": value,
        "notes": notes,
    })

add("cpsc2018_external_validation", "Valid AUROC", "CPSC 2018 external v2.1", "Feature baseline", "macro_auroc_valid", baseline.get("macro_auroc_valid"), "Valid labels: NORM/STTC/CD; MI/HYP excluded due to zero support.")
add("cpsc2018_external_validation", "Valid AUPRC", "CPSC 2018 external v2.1", "Feature baseline", "macro_auprc_valid", baseline.get("macro_auprc_valid"), "Valid labels: NORM/STTC/CD.")
add("cpsc2018_external_validation", "Valid Macro-F1", "CPSC 2018 external v2.1", "Feature baseline", "macro_f1_valid", baseline.get("macro_f1_valid"), "Valid labels: NORM/STTC/CD.")
add("cpsc2018_external_validation", "Valid sensitivity", "CPSC 2018 external v2.1", "Feature baseline", "macro_sensitivity_valid", baseline.get("macro_sensitivity_valid"), "Valid labels: NORM/STTC/CD.")

add("cpsc2018_external_validation", "Valid AUROC", "CPSC 2018 external v2.1", "InceptionTime", "macro_auroc_valid", deep_valid.get("macro_auroc_valid"), "Valid labels: NORM/STTC/CD; CPU inference.")
add("cpsc2018_external_validation", "Valid AUPRC", "CPSC 2018 external v2.1", "InceptionTime", "macro_auprc_valid", deep_valid.get("macro_auprc_valid"), "Valid labels: NORM/STTC/CD.")
add("cpsc2018_external_validation", "Valid Macro-F1", "CPSC 2018 external v2.1", "InceptionTime", "macro_f1_valid", deep_valid.get("macro_f1_valid"), "Valid labels: NORM/STTC/CD.")
add("cpsc2018_external_validation", "Valid sensitivity", "CPSC 2018 external v2.1", "InceptionTime", "macro_sensitivity_valid", deep_valid.get("macro_sensitivity_valid"), "Valid labels: NORM/STTC/CD.")
add("cpsc2018_external_validation", "Latency", "CPSC 2018 external v2.1", "InceptionTime", "inference_latency_ms_per_record", deep.get("inference_latency_ms_per_record"), "ms/record on CPU.")

df2 = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
df2.to_csv(results_csv, index=False)

# Append markdown addendum
addendum = f"""

---

## 11. CPSC 2018 External Validation Addendum

CPSC 2018 was added as a second external stress-test dataset after Georgia external v2.1. Download integrity passed with 6,877 paired records.

### CPSC 2018 label coverage under harmonization v2.1

- NORM: `918`
- STTC: `1087`
- CD: `2797`
- MI: `0`
- HYP: `0`

Because MI and HYP have zero mapped positive support under the current harmonization, CPSC 2018 metrics are reported using valid-label macro over NORM, STTC, and CD only.

### CPSC 2018 external results

| Model | Valid labels | AUROC | AUPRC | Macro-F1 | Precision | Sensitivity | Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Feature baseline | NORM/STTC/CD | {clean(baseline.get("macro_auroc_valid"))} | {clean(baseline.get("macro_auprc_valid"))} | {clean(baseline.get("macro_f1_valid"))} | {clean(baseline.get("macro_precision_valid"))} | {clean(baseline.get("macro_sensitivity_valid"))} | NA |
| InceptionTime | NORM/STTC/CD | {clean(deep_valid.get("macro_auroc_valid"))} | {clean(deep_valid.get("macro_auprc_valid"))} | {clean(deep_valid.get("macro_f1_valid"))} | {clean(deep_valid.get("macro_precision_valid"))} | {clean(deep_valid.get("macro_sensitivity_valid"))} | {clean(deep.get("inference_latency_ms_per_record"))} ms/record |

Interpretation: InceptionTime substantially outperforms the feature baseline on CPSC 2018 valid-label external testing.

---

## 12. Region Mapper v2.3 Addendum

Region Mapper v2.3 was added to reduce lateral-region bias in 3D/4D heart visualization.

Key changes:

- Region evidence is normalized by the number of leads per region.
- Ambiguous region assignments are allowed to become `uncertain`.
- Class-aware priors are used for STTC, MI, CD, and HYP.
- Lateral dominance from lead-count advantage is reduced.

Release artifacts:

- `region_mapping_v23/region_mapper_v23.py`
- `region_mapping_v23/region_mapper_v23_sanity_cases.csv`
- `region_mapping_v23/region_mapper_v23_summary.json`

---

## 13. Updated Next Work

1. Integrate `region_mapper_v23.py` into the dashboard 3D/4D heart map.
2. Add CPSC 2018 rows into paper-ready figure/table generation.
3. Improve harmonization for MI/HYP support across additional datasets.
4. Add prospective/real-time 12-lead ECG demo pipeline.
"""

text = summary_md.read_text(encoding="utf-8")
if "## 11. CPSC 2018 External Validation Addendum" not in text:
    text += addendum
summary_md.write_text(text, encoding="utf-8")

# Rebuild manifest
if zip_path.exists():
    zip_path.unlink()

files = []
for p in sorted(release.rglob("*")):
    if not p.is_file():
        continue
    if p.name == "release_rc1.zip":
        continue
    rel = p.relative_to(release).as_posix()
    st = p.stat()
    files.append({
        "path": rel,
        "size_bytes": int(st.st_size),
        "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(p),
    })

manifest = {}
if manifest_json.exists():
    manifest = read_json(manifest_json)

manifest.update({
    "project": "CardioTwin-AI 12L",
    "release": "v2.4 RC1 addendum",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "files_indexed": len(files),
    "files": files,
    "cpsc2018_external_validation": {
        "baseline_valid_metrics": baseline,
        "inceptiontime_valid_metrics": deep_valid,
        "inceptiontime_latency_ms_per_record": deep.get("inference_latency_ms_per_record"),
        "valid_labels": ["NORM", "STTC", "CD"],
        "excluded_labels": ["MI", "HYP"],
    },
    "region_mapping_v23": {
        "status": "added_to_release",
        "purpose": "reduce lateral-bias and allow uncertain region mapping."
    }
})
manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

# Rebuild ZIP
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(release.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == zip_path.resolve():
            continue
        if p.name.startswith("~$"):
            continue
        z.write(p, (Path("release_rc1") / p.relative_to(release)).as_posix())

print("DONE: Updated release with CPSC 2018 + Region Mapper v2.3")
print("Summary:", summary_md)
print("Results:", results_csv)
print("Manifest:", manifest_json)
print("ZIP:", zip_path)
print("ZIP size MB:", f"{zip_path.stat().st_size / 1024 / 1024:.2f}")
