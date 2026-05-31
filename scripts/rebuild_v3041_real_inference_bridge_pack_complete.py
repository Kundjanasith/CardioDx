from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"
OUT = ART / "unified_demo_v304"

OUT.mkdir(parents=True, exist_ok=True)
RELEASE.mkdir(parents=True, exist_ok=True)

smoke_path = OUT / "real_inference_smoke_test.json"
latest_path = OUT / "latest_v304_unified_result.json"

smoke = {}
latest = {}

if smoke_path.exists():
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))

if latest_path.exists():
    latest = json.loads(latest_path.read_text(encoding="utf-8"))

def normalize_result(obj):
    if not isinstance(obj, dict):
        return {}
    if "result" in obj and isinstance(obj["result"], dict):
        return obj["result"]
    return obj

smoke_result = normalize_result(smoke)
latest_result = normalize_result(latest)

def summarize_result(result):
    return {
        "inference_mode": result.get("inference_mode"),
        "model_loaded": result.get("model_meta", {}).get("loaded"),
        "positive_labels": result.get("positive_labels"),
        "abnormal_positive_labels": result.get("abnormal_positive_labels"),
        "recommendation": result.get("recommendation"),
        "threshold_source": result.get("threshold_source"),
        "region_mapper_used": result.get("region_mapper_meta", {}).get("used"),
        "sqi": result.get("sqi"),
    }

smoke_summary = summarize_result(smoke_result)
latest_summary = summarize_result(latest_result)

report_md = OUT / "REAL_INFERENCE_BRIDGE_REPORT.md"

report_text = f"""# CardioTwin-AI v3.0.4.1 Real Inference Bridge Report

Created: {datetime.now(timezone.utc).isoformat()}

## Purpose

This pack connects uploaded/replay ECG data to the frozen CardioTwin-AI v2.7 InceptionTime safety model, threshold profiles, Region Mapper v2.3, anatomical-style heart map, and unified JSON/HTML export.

## Confirmed Runtime Path

uploaded ECG / replay ECG
-> preprocess / resample
-> inceptiontime_v21_safety.pt
-> threshold_profiles_deep.json
-> region_mapper_v23
-> anatomical-style heart map
-> unified export report

## Smoke Test Summary

{json.dumps(smoke_summary, indent=2, ensure_ascii=False)}

## Latest Dashboard Runtime Summary

{json.dumps(latest_summary, indent=2, ensure_ascii=False)}

## Current Interpretation Example

If the screening profile flags HYP with probability slightly above threshold, interpret it as:

screening-positive for possible HYP pattern
-> doctor review recommended

It is not a final diagnosis.

## Claim Boundary

This is a research-use preliminary screening bridge. It is not final diagnosis and not clinical deployment.

High-risk, uncertain, low-SQI, or abnormal screening-positive cases require qualified human review.
"""

report_md.write_text(report_text, encoding="utf-8")

ZIP = RELEASE / "cardiotwin_v3_0_4_real_inference_bridge_pack.zip"
MANIFEST = RELEASE / "cardiotwin_v3_0_4_real_inference_bridge_manifest.json"

files = [
    ROOT / "src" / "cardiotwin" / "runtime" / "v304_real_inference_bridge.py",
    ROOT / "src" / "cardiotwin" / "runtime" / "__init__.py",
    ROOT / "apps" / "streamlit_cardiotwin_unified_v304_real_inference.py",
    ROOT / "scripts" / "smoke_test_v304_real_inference_bridge.py",
    ROOT / "scripts" / "patch_v304_bridge_v3041_model_region.py",
    ROOT / "scripts" / "create_v304_app_now.py",
    ART / "unified_demo_v304" / "real_inference_smoke_test.json",
    ART / "unified_demo_v304" / "latest_v304_unified_result.json",
    ART / "unified_demo_v304" / "REAL_INFERENCE_BRIDGE_REPORT.md",
]

files = [p for p in files if p.exists() and p.is_file()]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "project": "CardioTwin-AI",
    "version": "v3.0.4.1 Real Inference Bridge Pack",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Bridge uploaded/replay ECG to frozen v2.7 InceptionTime safety model, threshold profiles, region mapper v2.3, anatomical heart map, and unified export.",
    "recommended_command": "& $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v304_real_inference.py --server.port 8512",
    "claim_boundary": "Research-use preliminary screening support. Not final diagnosis and not clinical deployment.",
    "smoke_test_summary": smoke_summary,
    "latest_dashboard_summary": latest_summary,
    "files_indexed": len(files),
    "files": [
        {
            "path": p.as_posix(),
            "size_bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        }
        for p in files
    ],
}

MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

if ZIP.exists():
    ZIP.unlink()

with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.as_posix())
    z.write(MANIFEST, MANIFEST.as_posix())

print("DONE: rebuilt v3.0.4.1 real inference bridge pack")
print("ZIP:", ZIP)
print("ZIP size MB:", f"{ZIP.stat().st_size / 1024 / 1024:.2f}")
print("MANIFEST:", MANIFEST)
print("files_indexed:", manifest["files_indexed"])
print("smoke_test_summary:", manifest["smoke_test_summary"])
print("latest_dashboard_summary:", manifest["latest_dashboard_summary"])
print("included_report:", str(report_md), report_md.exists())
