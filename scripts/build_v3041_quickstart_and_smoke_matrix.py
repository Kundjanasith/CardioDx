from pathlib import Path
import json
import csv
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"
OUT = ART / "unified_demo_v304"

RELEASE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

quickstart = RELEASE / "CARDIOTWIN_V3041_QUICKSTART.md"

quickstart.write_text(f"""# CardioTwin-AI v3.0.4.1 Quickstart

Created: {datetime.now(timezone.utc).isoformat()}

## Release Name

CardioTwin-AI v3.0.4.1 Complete Runtime Release Bundle

## Main Bundle

`artifacts/release_rc1/cardiotwin_v3_0_4_1_complete_runtime_release_bundle.zip`

## Purpose

This release combines:

1. v2.7 RC1 frozen 12-lead AI/safety/export core
2. v2.8 BeatScope benchmark add-on
3. v3.0.3 unified demo dashboard
4. v3.0.4.1 real inference bridge

## Claim Boundary

Research-use preliminary ECG screening, visual explanation, and referral-support demo.

This is not final diagnosis and not clinical deployment.

## Environment Setup

Run in PowerShell:

    cd C:\\Users\\mrkit\\Downloads\\cardiotwin_ai_12l

    $PY = "C:\\venvs\\cardiotwin_v25\\Scripts\\python.exe"
    $env:PYTHONPATH = "$PWD\\src"
    $env:MPLBACKEND = "Agg"

## Open Unified Demo Dashboard

    & $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v303.py --server.port 8511

Open:

    http://localhost:8511

## Open Real Inference Bridge

    & $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v304_real_inference.py --server.port 8512

Open:

    http://localhost:8512

## Recommended First Test

Use:

    Input mode = WFDB path
    WFDB .hea path = data/raw/cinc2020/training/georgia/g1/E00001.hea
    Safety profile = screening
    Device = cpu

Expected:

    inference_mode = real_v2_7_torch_model
    model_loaded = True
    threshold_source = thresholds:screening
    region_mapper_used = True

## Interpretation Rule

If a class is screening-positive, interpret as:

    possible screening flag
    -> doctor review recommended

Do not interpret as confirmed diagnosis.

## Main Output Files

- `cardiotwin_v3_0_4_1_complete_runtime_release_bundle.zip`
- `cardiotwin_v3_0_4_1_complete_runtime_release_manifest.json`
- `cardiotwin_v3_0_4_real_inference_bridge_pack.zip`
- `cardiotwin_v3_0_4_real_inference_bridge_manifest.json`
- `CARDIOTWIN_V3041_QUICKSTART.md`

## Next Recommended Work

1. Run smoke matrix on multiple WFDB cases
2. Add locked external validation v3.1 with MIMIC-IV-ECG or KURIAS
3. Improve anatomical heart mesh in v3.0.5
4. Prepare mentor/demo slide deck and paper-ready methods update
""", encoding="utf-8")

# Build smoke test matrix from available Georgia / CPSC examples.
candidate_paths = []

for root in [
    Path("data/raw/cinc2020/training/georgia"),
    Path("data/raw/cinc2020/training/cpsc_2018"),
    Path("data/raw/cinc2020/training/cpsc_2018_extra"),
]:
    if root.exists():
        candidate_paths.extend(sorted(root.rglob("*.hea"))[:5])

# Keep max 12 to avoid long runtime.
candidate_paths = candidate_paths[:12]

rows = []

try:
    from cardiotwin.runtime.v304_real_inference_bridge import (
        load_wfdb_hea_mat,
        run_v304_real_inference,
    )

    for idx, hea in enumerate(candidate_paths, start=1):
        row = {
            "case_no": idx,
            "hea_path": str(hea),
            "status": "pending",
            "inference_mode": "",
            "model_loaded": "",
            "sqi": "",
            "positive_labels": "",
            "abnormal_positive_labels": "",
            "recommendation": "",
            "threshold_source": "",
            "region_mapper_used": "",
            "error": "",
        }

        try:
            x, fs, meta = load_wfdb_hea_mat(hea)
            result = run_v304_real_inference(
                x_raw=x,
                fs=fs,
                model_path="artifacts/models/inceptiontime_v21_safety.pt",
                threshold_path="artifacts/deep_safety_v21/threshold_profiles_deep.json",
                profile="screening",
                device="cpu",
                source_meta=meta,
            )

            row.update({
                "status": "ok",
                "inference_mode": result.get("inference_mode"),
                "model_loaded": result.get("model_meta", {}).get("loaded"),
                "sqi": result.get("sqi"),
                "positive_labels": "|".join(result.get("positive_labels", [])),
                "abnormal_positive_labels": "|".join(result.get("abnormal_positive_labels", [])),
                "recommendation": result.get("recommendation"),
                "threshold_source": result.get("threshold_source"),
                "region_mapper_used": result.get("region_mapper_meta", {}).get("used"),
            })

        except Exception as e:
            row["status"] = "error"
            row["error"] = repr(e)

        rows.append(row)

except Exception as e:
    rows.append({
        "case_no": 0,
        "hea_path": "",
        "status": "bridge_import_error",
        "inference_mode": "",
        "model_loaded": "",
        "sqi": "",
        "positive_labels": "",
        "abnormal_positive_labels": "",
        "recommendation": "",
        "threshold_source": "",
        "region_mapper_used": "",
        "error": repr(e),
    })

csv_path = OUT / "smoke_matrix_v3041.csv"
fieldnames = [
    "case_no",
    "hea_path",
    "status",
    "inference_mode",
    "model_loaded",
    "sqi",
    "positive_labels",
    "abnormal_positive_labels",
    "recommendation",
    "threshold_source",
    "region_mapper_used",
    "error",
]

with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

ok_rows = [r for r in rows if r["status"] == "ok"]
real_rows = [r for r in ok_rows if r["inference_mode"] == "real_v2_7_torch_model"]
loaded_rows = [r for r in ok_rows if str(r["model_loaded"]) == "True"]
region_rows = [r for r in ok_rows if str(r["region_mapper_used"]) == "True"]

summary = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "n_cases": len(rows),
    "n_ok": len(ok_rows),
    "n_error": len(rows) - len(ok_rows),
    "n_real_v2_7_torch_model": len(real_rows),
    "n_model_loaded_true": len(loaded_rows),
    "n_region_mapper_used_true": len(region_rows),
    "all_ok": len(ok_rows) == len(rows) and len(rows) > 0,
    "csv_path": str(csv_path),
    "quickstart_path": str(quickstart),
}

summary_path = OUT / "smoke_matrix_v3041_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print("DONE: Quickstart + smoke matrix created")
print("QUICKSTART:", quickstart)
print("SMOKE CSV:", csv_path)
print("SMOKE SUMMARY:", summary_path)
print(json.dumps(summary, indent=2, ensure_ascii=False))
