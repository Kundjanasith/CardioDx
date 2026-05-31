from pathlib import Path
import json
from datetime import datetime, timezone

OUT = Path("artifacts/locked_external_validation_v30")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "MIMIC_IV_ECG": Path("data/raw/mimic_iv_ecg"),
    "KURIAS_ECG": Path("data/raw/kurias_ecg"),
}

def count_files(root: Path, suffixes):
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)

report = {
    "version": "locked_external_readiness_v30",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "datasets": {},
    "recommendation": "",
}

for name, root in DATASETS.items():
    info = {
        "path": str(root),
        "exists": root.exists(),
        "hea_count": count_files(root, {".hea"}),
        "mat_count": count_files(root, {".mat"}),
        "csv_count": count_files(root, {".csv"}),
        "dat_count": count_files(root, {".dat"}),
        "ready_for_adapter": False,
        "notes": "",
    }
    if info["exists"] and (info["hea_count"] > 0 or info["csv_count"] > 0 or info["dat_count"] > 0):
        info["ready_for_adapter"] = True
        info["notes"] = "Dataset files detected. Next step: implement dataset-specific waveform/report adapter and label mapping."
    else:
        info["notes"] = "Dataset not found or no recognizable files detected."
    report["datasets"][name] = info

if report["datasets"]["MIMIC_IV_ECG"]["ready_for_adapter"]:
    report["recommendation"] = "Proceed with MIMIC-IV-ECG adapter and locked label mapping."
elif report["datasets"]["KURIAS_ECG"]["ready_for_adapter"]:
    report["recommendation"] = "Proceed with KURIAS-ECG adapter and ontology mapping."
else:
    report["recommendation"] = "Place MIMIC-IV-ECG or KURIAS-ECG files under data/raw before running locked external validation."

(OUT / "locked_external_dataset_readiness_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps(report, indent=2, ensure_ascii=False))
