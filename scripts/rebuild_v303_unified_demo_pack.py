from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"

ZIP = RELEASE / "cardiotwin_v3_0_3_unified_demo_pack.zip"
MANIFEST = ART / "v303_unified_demo_pack_manifest.json"

include_roots = [
    ART / "locked_external_validation_v30",
    ART / "prospective_pilot_v30",
    ART / "human_review_v30",
    ART / "realtime_demo_v30",
    ART / "risk_management_v30",
    ART / "cost_effectiveness_v30",
    ART / "pitch_pack_v30",
    ART / "report_templates_v30",
    ART / "product_readiness_v30",
    ART / "unified_demo_v302",
]

extra_files = [
    ROOT / "apps" / "streamlit_realtime_replay_v30.py",
    ROOT / "apps" / "streamlit_realtime_replay_v301_heart.py",
    ROOT / "apps" / "streamlit_cardiotwin_unified_v302.py",
    ROOT / "apps" / "streamlit_cardiotwin_unified_v303.py",
    ROOT / "scripts" / "run_locked_external_readiness_v30.py",
    ROOT / "scripts" / "patch_unified_dashboard_v303_demo_polish.py",
]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = []

for root in include_roots:
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files.append(p)

for p in extra_files:
    if p.exists():
        files.append(p)

manifest = {
    "project": "CardioTwin-AI",
    "version": "v3.0.3 Unified Clinical Demo Pack",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "positioning": (
        "Integrated world-class demo platform connecting research evidence, "
        "AI inference status, visual explanation, clinical workflow, and real-time presentation."
    ),
    "relationship_to_previous_releases": {
        "v2.7": "frozen 12-lead record-level AI/safety/export release",
        "v2.8": "frozen BeatScope beat-level benchmark add-on",
        "v3.0": "clinical pilot readiness pack",
        "v3.0.1": "realtime replay + heart visual demo",
        "v3.0.2": "unified demo dashboard integration layer",
        "v3.0.3": "demo-polished unified dashboard with clean balanced mode calibration",
    },
    "claim_boundary": (
        "Research-use preliminary screening and visual explanation demo. "
        "Not final diagnosis and not clinical deployment. "
        "Strict frozen-model inference remains under the v2.7 dashboard/core."
    ),
    "included_apps": [
        "apps/streamlit_realtime_replay_v30.py",
        "apps/streamlit_realtime_replay_v301_heart.py",
        "apps/streamlit_cardiotwin_unified_v302.py",
        "apps/streamlit_cardiotwin_unified_v303.py",
    ],
    "recommended_demo_app": "apps/streamlit_cardiotwin_unified_v303.py",
    "recommended_demo_command": (
        "& $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v303.py --server.port 8511"
    ),
    "files_indexed": len(files),
    "files": [
        {
            "path": p.as_posix(),
            "size_bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        }
        for p in sorted(files)
    ],
}

MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

if ZIP.exists():
    ZIP.unlink()

with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):
        z.write(p, p.as_posix())
    z.write(MANIFEST, "artifacts/v303_unified_demo_pack_manifest.json")

print("DONE: rebuilt v3.0.3 unified demo pack")
print("ZIP:", ZIP)
print("ZIP size MB:", f"{ZIP.stat().st_size / 1024 / 1024:.2f}")
print("MANIFEST:", MANIFEST)
print("files_indexed:", manifest["files_indexed"])
print("recommended_demo_app:", manifest["recommended_demo_app"])
