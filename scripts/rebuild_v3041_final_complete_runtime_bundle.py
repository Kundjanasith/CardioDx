from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"
OUT = ART / "unified_demo_v304"

OUT_ZIP = RELEASE / "cardiotwin_v3_0_4_1_complete_runtime_release_bundle.zip"
OUT_MANIFEST = RELEASE / "cardiotwin_v3_0_4_1_complete_runtime_release_manifest.json"

candidates = [
    RELEASE / "release_rc1.zip",
    RELEASE / "cardiotwin_ai_12l_v2_7_rc1_release.zip",
    RELEASE / "cardiotwin_ai_12l_v2_7_rc1_manifest.json",
    RELEASE / "cardiotwin_ai_12l_v2_7_rc1_research_summary.md",

    RELEASE / "cardiotwin_beatscope_v2_8_full_addon.zip",
    RELEASE / "cardiotwin_beatscope_v2_8_full_manifest.json",

    RELEASE / "cardiotwin_v3_0_3_complete_unified_release_bundle.zip",
    RELEASE / "cardiotwin_v3_0_3_complete_unified_release_manifest.json",

    RELEASE / "cardiotwin_v3_0_4_real_inference_bridge_pack.zip",
    RELEASE / "cardiotwin_v3_0_4_real_inference_bridge_manifest.json",

    RELEASE / "CARDIOTWIN_V3041_QUICKSTART.md",
    OUT / "smoke_matrix_v3041.csv",
    OUT / "smoke_matrix_v3041_summary.json",
    OUT / "REAL_INFERENCE_BRIDGE_REPORT.md",
    OUT / "real_inference_smoke_test.json",
    OUT / "latest_v304_unified_result.json",
]

files = [p for p in candidates if p.exists() and p.is_file()]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

smoke_summary_path = OUT / "smoke_matrix_v3041_summary.json"
smoke_summary = {}
if smoke_summary_path.exists():
    smoke_summary = json.loads(smoke_summary_path.read_text(encoding="utf-8"))

manifest = {
    "project": "CardioTwin-AI",
    "version": "v3.0.4.1 Complete Runtime Release Bundle FINAL",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": (
        "Complete handoff bundle combining frozen v2.7 12-lead AI/safety core, "
        "v2.8 BeatScope benchmark add-on, v3.0.3 unified demo dashboard, "
        "v3.0.4.1 real inference bridge, quickstart guide, and 12-case smoke matrix."
    ),
    "recommended_demo_commands": {
        "unified_demo_v303": "& $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v303.py --server.port 8511",
        "real_inference_bridge_v304": "& $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v304_real_inference.py --server.port 8512"
    },
    "claim_boundary": (
        "Research-use preliminary ECG screening, visual explanation, and referral-support demo. "
        "Not final diagnosis and not clinical deployment."
    ),
    "release_components": {
        "v2.7_rc1": "Frozen 12-lead record-level ECG AI + safety calibration + export core",
        "v2.8_beatscope": "Beat-level ECG morphology benchmark + MIT-BIH/PTBDB + transfer learning add-on",
        "v3.0.3_unified_demo": "Integrated presentation dashboard",
        "v3.0.4.1_real_inference_bridge": "Runtime bridge from ECG input to frozen v2.7 model + thresholds + region mapper",
        "quickstart": "User-facing quickstart for opening dashboards and running first test",
        "smoke_matrix": "12-case runtime smoke matrix verifying model load + real inference + region mapper"
    },
    "smoke_matrix_summary": smoke_summary,
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

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.as_posix())
    z.write(OUT_MANIFEST, OUT_MANIFEST.as_posix())

print("DONE: FINAL complete runtime release bundle rebuilt")
print("ZIP:", OUT_ZIP)
print("ZIP size MB:", f"{OUT_ZIP.stat().st_size / 1024 / 1024:.2f}")
print("MANIFEST:", OUT_MANIFEST)
print("files_indexed:", manifest["files_indexed"])
print("smoke_matrix_summary:", manifest["smoke_matrix_summary"])
