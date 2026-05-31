from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

release = Path("artifacts/release_rc1")
summary_md = release / "CARDIOTWIN_RC1_RESEARCH_SUMMARY.md"
manifest_json = release / "release_manifest.json"
zip_path = release / "release_rc1.zip"

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Fix mojibake in summary if present
if summary_md.exists():
    text = summary_md.read_text(encoding="utf-8")
    text = text.replace("โ€”", "-")
    summary_md.write_text(text, encoding="utf-8")

# Load manifest
manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

# Fix title
manifest["title"] = "CardioTwin-AI 12L v2.4 RC1 - External-Validated + Safety-Calibrated Deep ECG Screening Prototype"
manifest["release"] = "v2.4 RC1"
manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

# Update high-level metrics with strongest current external result
manifest["high_level_metrics"]["best_external_model"] = "InceptionTime"
manifest["high_level_metrics"]["best_external_dataset"] = "CPSC 2018 external v2.1"
manifest["high_level_metrics"]["best_external_valid_auroc"] = 0.8535993614864067
manifest["high_level_metrics"]["best_external_valid_auprc"] = 0.6883273453878078
manifest["high_level_metrics"]["best_external_valid_macro_f1"] = 0.6168531362037758
manifest["high_level_metrics"]["note"] = "CPSC 2018 best external metrics use valid-label macro over NORM/STTC/CD; MI/HYP excluded due to zero mapped support."

# Rebuild file index, excluding zip and manifest itself to avoid stale self-hash
files = []
for p in sorted(release.rglob("*")):
    if not p.is_file():
        continue
    if p.name in {"release_rc1.zip", "release_manifest.json"}:
        continue
    if p.name.startswith("~$"):
        continue
    st = p.stat()
    files.append({
        "path": p.relative_to(release).as_posix(),
        "size_bytes": int(st.st_size),
        "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(p),
    })

manifest["files_indexed"] = len(files)
manifest["files"] = files
manifest["manifest_note"] = "release_manifest.json and release_rc1.zip are excluded from file self-indexing to avoid stale self-referential hashes."

manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

# Rebuild ZIP
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(release.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == zip_path.resolve():
            continue
        if p.name.startswith("~$"):
            continue
        z.write(p, (Path("release_rc1") / p.relative_to(release)).as_posix())

print("DONE: Cleaned manifest title, updated high-level metrics, rebuilt ZIP")
print("Manifest:", manifest_json)
print("ZIP:", zip_path)
print("ZIP size MB:", f"{zip_path.stat().st_size / 1024 / 1024:.2f}")
