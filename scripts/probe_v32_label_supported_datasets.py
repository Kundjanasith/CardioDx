from pathlib import Path
import json
import csv
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
OUT = ART / "label_supported_external_validation_v32"
OUT.mkdir(parents=True, exist_ok=True)

created = datetime.now(timezone.utc).isoformat()

candidates = {
    "full_mimic_iv_ecg": {
        "root": Path("data/raw/mimic_iv_ecg"),
        "expected": "WFDB waveform files + report/diagnosis metadata"
    },
    "kurias_ecg": {
        "root": Path("data/raw/kurias_ecg"),
        "expected": "12-lead ECG waveform files + diagnosis/ontology metadata"
    }
}

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def scan_csv_metadata(root):
    results = []
    if not root.exists():
        return results

    for p in sorted(root.rglob("*.csv"))[:50]:
        item = {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "columns": [],
            "n_preview_rows": 0,
            "likely_id_columns": [],
            "likely_label_columns": [],
            "likely_report_columns": [],
        }

        try:
            with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                item["columns"] = cols

                for i, _ in enumerate(reader):
                    if i >= 20:
                        break
                    item["n_preview_rows"] += 1

            for c in cols:
                lc = c.lower()
                if lc in ["subject_id", "study_id", "record_id", "ecg_id", "file_name", "path"]:
                    item["likely_id_columns"].append(c)
                if any(k in lc for k in ["diagnosis", "label", "snomed", "code", "statement", "class", "category", "ontology"]):
                    item["likely_label_columns"].append(c)
                if any(k in lc for k in ["report", "interpret", "comment", "text", "finding"]):
                    item["likely_report_columns"].append(c)

        except Exception as e:
            item["error"] = repr(e)

        results.append(item)

    return results

dataset_reports = {}

for name, cfg in candidates.items():
    root = cfg["root"]

    hea_files = sorted(root.rglob("*.hea")) if root.exists() else []
    dat_files = sorted(root.rglob("*.dat")) if root.exists() else []
    mat_files = sorted(root.rglob("*.mat")) if root.exists() else []
    csv_files = sorted(root.rglob("*.csv")) if root.exists() else []
    txt_files = sorted(root.rglob("*.txt")) if root.exists() else []
    tsv_files = sorted(root.rglob("*.tsv")) if root.exists() else []

    metadata_scan = scan_csv_metadata(root)

    likely_label_csvs = [
        x for x in metadata_scan
        if x.get("likely_label_columns") or x.get("likely_report_columns")
    ]

    waveform_ready = len(hea_files) > 0 and (len(dat_files) > 0 or len(mat_files) > 0)
    label_ready = len(likely_label_csvs) > 0

    if waveform_ready and label_ready:
        readiness = "ready_for_label_supported_validation_candidate"
    elif waveform_ready and not label_ready:
        readiness = "waveform_ready_but_label_metadata_missing_or_unidentified"
    elif not waveform_ready and label_ready:
        readiness = "label_metadata_found_but_waveforms_missing_or_unidentified"
    else:
        readiness = "not_ready"

    dataset_reports[name] = {
        "root": str(root),
        "expected": cfg["expected"],
        "exists": root.exists(),
        "counts": {
            "hea_count": len(hea_files),
            "dat_count": len(dat_files),
            "mat_count": len(mat_files),
            "csv_count": len(csv_files),
            "txt_count": len(txt_files),
            "tsv_count": len(tsv_files),
        },
        "waveform_ready": waveform_ready,
        "label_metadata_candidate_found": label_ready,
        "readiness": readiness,
        "metadata_scan": metadata_scan,
        "likely_label_or_report_csvs": likely_label_csvs,
        "example_hea_files": [str(p) for p in hea_files[:5]],
        "example_csv_files": [str(p) for p in csv_files[:5]],
    }

summary = {
    "project": "CardioTwin-AI",
    "version": "v3.2.1 dataset readiness probe",
    "created_at_utc": created,
    "purpose": "Check local availability of Full MIMIC-IV-ECG and KURIAS for label-supported external validation.",
    "datasets": dataset_reports,
    "recommendation": "Use the first dataset with waveform_ready=true and label_metadata_candidate_found=true. If none are ready, download/prepare dataset metadata before running performance metrics.",
    "claim_boundary": "Readiness probe only. No validation metrics computed."
}

out_json = OUT / "dataset_readiness_probe_v321.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

out_md = OUT / "DATASET_READINESS_PROBE_v321.md"
lines = []
lines.append("# CardioTwin-AI v3.2.1 Dataset Readiness Probe")
lines.append("")
lines.append(f"Created: {created}")
lines.append("")
for name, report in dataset_reports.items():
    lines.append(f"## {name}")
    lines.append("")
    lines.append(f"- Root: `{report['root']}`")
    lines.append(f"- Exists: `{report['exists']}`")
    lines.append(f"- Readiness: `{report['readiness']}`")
    lines.append(f"- HEA: `{report['counts']['hea_count']}`")
    lines.append(f"- DAT: `{report['counts']['dat_count']}`")
    lines.append(f"- MAT: `{report['counts']['mat_count']}`")
    lines.append(f"- CSV: `{report['counts']['csv_count']}`")
    lines.append(f"- Label/report metadata candidate found: `{report['label_metadata_candidate_found']}`")
    lines.append("")
lines.append("## Boundary")
lines.append("")
lines.append("This is a readiness probe only. No AUROC/AUPRC/F1 metrics are computed here.")
out_md.write_text("\n".join(lines), encoding="utf-8")

print("DONE: v3.2.1 dataset readiness probe")
print("JSON:", out_json)
print("MD:", out_md)
print(json.dumps({
    k: {
        "exists": v["exists"],
        "readiness": v["readiness"],
        "counts": v["counts"],
        "label_metadata_candidate_found": v["label_metadata_candidate_found"],
    }
    for k, v in dataset_reports.items()
}, indent=2, ensure_ascii=False))
