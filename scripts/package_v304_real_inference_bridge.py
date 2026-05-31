from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"

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
    ART / "unified_demo_v304" / "REAL_INFERENCE_BRIDGE_REPORT.md",
]

files = [p for p in files if p.exists() and p.is_file()]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

smoke = {}
smoke_path = ART / "unified_demo_v304" / "real_inference_smoke_test.json"
if smoke_path.exists():
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))

manifest = {
    "project": "CardioTwin-AI",
    "version": "v3.0.4.1 Real Inference Bridge Pack",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Bridge uploaded/replay ECG to frozen v2.7 InceptionTime safety model, threshold profiles, region mapper v2.3, anatomical heart map, and unified export.",
    "recommended_command": "& $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v304_real_inference.py --server.port 8512",
    "claim_boundary": "Research-use preliminary screening support. Not final diagnosis and not clinical deployment.",
    "smoke_test_summary": {
        "inference_mode": smoke.get("inference_mode"),
        "model_loaded": smoke.get("model_meta", {}).get("loaded"),
        "positive_labels": smoke.get("positive_labels"),
        "recommendation": smoke.get("recommendation"),
        "threshold_source": smoke.get("threshold_source"),
        "region_mapper_used": smoke.get("region_mapper_meta", {}).get("used"),
    },
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

print("DONE: v3.0.4.1 real inference bridge pack created")
print("ZIP:", ZIP)
print("ZIP size MB:", f"{ZIP.stat().st_size / 1024 / 1024:.2f}")
print("MANIFEST:", MANIFEST)
print("files_indexed:", manifest["files_indexed"])
print("smoke_test_summary:", manifest["smoke_test_summary"])
