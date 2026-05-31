from pathlib import Path
import shutil
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"
BACKUP_ROOT = ART / "release_backups"
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
backup_dir = BACKUP_ROOT / f"release_rc1_backup_{timestamp}"

if not RELEASE.exists():
    raise FileNotFoundError("artifacts/release_rc1 not found")

shutil.copytree(RELEASE, backup_dir)

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = [p for p in sorted(backup_dir.rglob("*")) if p.is_file()]

manifest = {
    "project": "CardioTwin-AI",
    "backup_type": "release_rc1_full_folder_backup",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": str(RELEASE),
    "backup_dir": str(backup_dir),
    "files_indexed": len(files),
    "files": [
        {
            "path": p.relative_to(backup_dir).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in files
    ],
}

manifest_path = backup_dir / "BACKUP_MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("DONE: release_rc1 backup completed")
print("BACKUP_DIR:", backup_dir)
print("MANIFEST:", manifest_path)
print("files_indexed:", manifest["files_indexed"])
