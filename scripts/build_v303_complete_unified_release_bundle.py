from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"

OUT_ZIP = RELEASE / "cardiotwin_v3_0_3_complete_unified_release_bundle.zip"
OUT_MANIFEST = RELEASE / "cardiotwin_v3_0_3_complete_unified_release_manifest.json"

candidates = [
    RELEASE / "release_rc1.zip",
    RELEASE / "cardiotwin_ai_12l_v2_7_rc1_release.zip",
    RELEASE / "cardiotwin_ai_12l_v2_7_rc1_manifest.json",
    RELEASE / "cardiotwin_ai_12l_v2_7_rc1_research_summary.md",
    RELEASE / "cardiotwin_beatscope_v2_8_full_addon.zip",
    RELEASE / "cardiotwin_beatscope_v2_8_full_manifest.json",
    RELEASE / "cardiotwin_v3_0_3_unified_demo_pack.zip",
    ART / "v303_unified_demo_pack_manifest.json",
]

files = []
for p in candidates:
    if p.exists() and p.is_file():
        files.append(p)

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "project": "CardioTwin-AI",
    "version": "v3.0.3 Complete Unified Release Bundle",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Complete handoff bundle combining frozen v2.7 core release, v2.8 BeatScope full add-on, and v3.0.3 unified demo integration pack.",
    "recommended_demo_command": "& $PY -m streamlit run apps\\streamlit_cardiotwin_unified_v303.py --server.port 8511",
    "claim_boundary": "Research-use preliminary screening and visual explanation demo. Not final diagnosis and not clinical deployment.",
    "bundle_note": "This complete bundle references frozen v2.7 model/core artifacts and v2.8 BeatScope evidence while keeping the v3.0.3 dashboard as an integration/presentation layer.",
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

print("DONE: complete unified release bundle created")
print("ZIP:", OUT_ZIP)
print("ZIP size MB:", f"{OUT_ZIP.stat().st_size / 1024 / 1024:.2f}")
print("MANIFEST:", OUT_MANIFEST)
print("files_indexed:", manifest["files_indexed"])
for f in manifest["files"]:
    print("-", f["path"], f["size_bytes"])
