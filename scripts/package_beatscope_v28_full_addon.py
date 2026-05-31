from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

SRC = Path("artifacts/release_rc1/heartbeat_benchmark_v28")
OUT = Path("artifacts/release_rc1")
ZIP = OUT / "cardiotwin_beatscope_v2_8_full_addon.zip"
MANIFEST = OUT / "cardiotwin_beatscope_v2_8_full_manifest.json"

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

best_mit = summary.get("best_mitbih_by_macro_f1", [{}])[0]
best_ptb = summary.get("best_ptbdb_by_macro_f1", [{}])[0]
transfer = summary.get("transfer_summary", {})
gains = transfer.get("gains", {})

manifest = {
    "project": "CardioTwin-AI BeatScope",
    "version": "v2.8 full add-on",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "relationship_to_v27": "Auxiliary beat-level benchmark add-on; does not modify frozen CardioTwin-AI v2.7 RC1.",
    "claim_boundary": "Beat-level segmented ECG benchmark. Do not mix with 12-lead record-level validation metrics.",
    "mode": summary.get("mode"),
    "device": summary.get("device"),
    "headline_results": {
        "mitbih_best_model": best_mit.get("model"),
        "mitbih_macro_f1": best_mit.get("macro_f1"),
        "mitbih_auroc_macro": best_mit.get("auroc_macro"),
        "mitbih_auprc_macro": best_mit.get("auprc_macro"),
        "ptbdb_best_model": best_ptb.get("model"),
        "ptbdb_macro_f1": best_ptb.get("macro_f1"),
        "ptbdb_auroc_macro": best_ptb.get("auroc_macro"),
        "ptbdb_auprc_macro": best_ptb.get("auprc_macro"),
        "transfer_balanced_accuracy_gain": gains.get("balanced_accuracy_gain"),
        "transfer_macro_f1_gain": gains.get("macro_f1_gain"),
        "transfer_auroc_macro_gain": gains.get("auroc_macro_gain"),
        "transfer_auprc_macro_gain": gains.get("auprc_macro_gain"),
    },
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

print("DONE: BeatScope v2.8 full add-on packaged")
print("ZIP:", ZIP)
print("ZIP size MB:", f"{ZIP.stat().st_size / 1024 / 1024:.2f}")
print("MANIFEST:", MANIFEST)
print("files_indexed:", len(files))
print("MIT-BIH best:", best_mit)
print("PTBDB best:", best_ptb)
print("Transfer gains:", gains)
