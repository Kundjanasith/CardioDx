from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"
OUT = ART / "locked_external_validation_v31"

RELEASE.mkdir(parents=True, exist_ok=True)

OUT_ZIP = RELEASE / "cardiotwin_v3_1_4_complete_runtime_plus_mimic_demo_bundle.zip"
OUT_MANIFEST = RELEASE / "cardiotwin_v3_1_4_complete_runtime_plus_mimic_demo_manifest.json"

candidates = [
    # v2.7 core
    RELEASE / "release_rc1.zip",
    RELEASE / "cardiotwin_ai_12l_v2_7_rc1_release.zip",
    RELEASE / "cardiotwin_ai_12l_v2_7_rc1_manifest.json",
    RELEASE / "cardiotwin_ai_12l_v2_7_rc1_research_summary.md",

    # v2.8 BeatScope
    RELEASE / "cardiotwin_beatscope_v2_8_full_addon.zip",
    RELEASE / "cardiotwin_beatscope_v2_8_full_manifest.json",

    # v3.0.3 unified demo
    RELEASE / "cardiotwin_v3_0_3_complete_unified_release_bundle.zip",
    RELEASE / "cardiotwin_v3_0_3_complete_unified_release_manifest.json",

    # v3.0.4.1 real inference bridge
    RELEASE / "cardiotwin_v3_0_4_real_inference_bridge_pack.zip",
    RELEASE / "cardiotwin_v3_0_4_real_inference_bridge_manifest.json",

    # v3.0.4.1 complete runtime bundle
    RELEASE / "cardiotwin_v3_0_4_1_complete_runtime_release_bundle.zip",
    RELEASE / "cardiotwin_v3_0_4_1_complete_runtime_release_manifest.json",

    # v3.1.2.1 MIMIC demo runtime dry-run
    RELEASE / "cardiotwin_v3_1_2_1_mimic_demo_runtime_dryrun_pack.zip",
    RELEASE / "cardiotwin_v3_1_2_1_mimic_demo_runtime_dryrun_manifest.json",

    # v3.1.3 audit
    RELEASE / "cardiotwin_v3_1_3_mimic_demo_label_report_audit_pack.zip",
    RELEASE / "cardiotwin_v3_1_3_mimic_demo_label_report_audit_manifest.json",

    # v3.1.4 final MIMIC Demo label-free runtime validation
    RELEASE / "cardiotwin_v3_1_4_mimic_demo_label_free_runtime_validation_pack.zip",
    RELEASE / "cardiotwin_v3_1_4_mimic_demo_label_free_runtime_validation_manifest.json",

    # key readable reports
    OUT / "MIMIC_DEMO_LABEL_FREE_RUNTIME_VALIDATION_FINAL_v314.md",
    OUT / "MIMIC_DEMO_LABEL_FREE_RUNTIME_VALIDATION_FINAL_v314.html",
    OUT / "mimic_demo_label_report_audit_v313.json",
    OUT / "mimic_demo_locked_dryrun_runtime_summary_v3121.json",
]

files = [p for p in candidates if p.exists() and p.is_file()]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

v314_manifest = RELEASE / "cardiotwin_v3_1_4_mimic_demo_label_free_runtime_validation_manifest.json"
v314 = {}
if v314_manifest.exists():
    v314 = json.loads(v314_manifest.read_text(encoding="utf-8"))

manifest = {
    "project": "CardioTwin-AI",
    "version": "v3.1.4 Complete Runtime + MIMIC Demo Label-free Validation Bundle",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": (
        "Complete handoff bundle combining v2.7 core, v2.8 BeatScope, v3.0.3 unified demo, "
        "v3.0.4.1 real inference bridge, and v3.1.4 MIMIC-IV-ECG Demo label-free runtime validation."
    ),
    "runtime_status": {
        "v3_0_4_1": "Frozen runtime release complete",
        "v3_1_4": "MIMIC-IV-ECG Demo label-free runtime validation complete",
    },
    "mimic_demo_result": v314.get("runtime_result", {}),
    "mimic_demo_claim_boundary": {
        "allowed_claim": v314.get("allowed_claim"),
        "disallowed_claim": v314.get("disallowed_claim"),
    },
    "recommended_demo_commands": {
        "unified_demo_v303": "& $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v303.py --server.port 8511",
        "real_inference_bridge_v304": "& $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v304_real_inference.py --server.port 8512",
    },
    "next_step": "Start v3.2 label-supported external validation using full MIMIC-IV-ECG or KURIAS.",
    "claim_boundary": "Research-use ECG screening, visual explanation, runtime validation, and external-validation preparation only. Not final diagnosis and not clinical deployment.",
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

print("DONE: v3.1.4 complete runtime + MIMIC Demo bundle created")
print("ZIP:", OUT_ZIP)
print("ZIP size MB:", f"{OUT_ZIP.stat().st_size / 1024 / 1024:.2f}")
print("MANIFEST:", OUT_MANIFEST)
print("files_indexed:", manifest["files_indexed"])
print("mimic_demo_result:", manifest["mimic_demo_result"])
