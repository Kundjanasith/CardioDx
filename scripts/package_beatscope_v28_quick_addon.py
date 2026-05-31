from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

SRC = Path("artifacts/release_rc1/heartbeat_benchmark_v28")
OUT = Path("artifacts/release_rc1")
ZIP = OUT / "cardiotwin_beatscope_v2_8_quick_addon.zip"
MANIFEST = OUT / "cardiotwin_beatscope_v2_8_quick_manifest.json"

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = []
for p in sorted(SRC.rglob("*")):
    if p.is_file():
        files.append({
            "path": p.relative_to(SRC).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

summary_path = SRC / "beatscope_v28_run_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

manifest = {
    "project": "CardioTwin-AI BeatScope",
    "version": "v2.8 quick add-on",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "relationship_to_v27": "Auxiliary beat-level benchmark add-on; does not modify frozen CardioTwin-AI v2.7 RC1.",
    "claim_boundary": "Beat-level segmented ECG benchmark. Do not mix with 12-lead record-level validation metrics.",
    "run_summary": summary,
    "files_indexed": len(files),
    "files": files,
}

MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

if ZIP.exists():
    ZIP.unlink()

with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(SRC.rglob("*")):
        if p.is_file():
            z.write(p, (Path("heartbeat_benchmark_v28") / p.relative_to(SRC)).as_posix())
    z.write(MANIFEST, MANIFEST.name)

print("DONE: BeatScope v2.8 quick add-on packaged")
print("ZIP:", ZIP)
print("ZIP size MB:", f"{ZIP.stat().st_size / 1024 / 1024:.2f}")
print("MANIFEST:", MANIFEST)
print("files_indexed:", len(files))
