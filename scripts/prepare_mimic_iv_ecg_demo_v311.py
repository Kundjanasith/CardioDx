from pathlib import Path
import argparse
import csv
import json
import zipfile
import tarfile
import hashlib
import re
from datetime import datetime, timezone

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x, default=None):
    try:
        return float(str(x).split("/")[0])
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        return int(float(str(x).split("/")[0]))
    except Exception:
        return default


def maybe_extract_archives(raw_dir):
    extracted = []
    archives = []
    for pattern in ["*.zip", "*.tar", "*.tar.gz", "*.tgz"]:
        archives.extend(raw_dir.rglob(pattern))

    for arc in sorted(set(archives)):
        try:
            if arc.suffix.lower() == ".zip":
                out_dir = arc.parent / arc.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(arc, "r") as z:
                    z.extractall(out_dir)
                extracted.append({"archive": str(arc), "out_dir": str(out_dir), "status": "ok_zip"})
            elif arc.name.endswith(".tar.gz") or arc.name.endswith(".tgz") or arc.suffix.lower() == ".tar":
                name = arc.name
                for suffix in [".tar.gz", ".tgz", ".tar"]:
                    if name.endswith(suffix):
                        name = name[: -len(suffix)]
                        break
                out_dir = arc.parent / name
                out_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(arc, "r:*") as t:
                    t.extractall(out_dir)
                extracted.append({"archive": str(arc), "out_dir": str(out_dir), "status": "ok_tar"})
        except Exception as e:
            extracted.append({"archive": str(arc), "out_dir": "", "status": "error", "error": repr(e)})

    return extracted


def parse_hea_header(hea_path):
    text = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not text:
        return {
            "status": "empty_header",
            "record_name": hea_path.stem,
            "n_sig": None,
            "fs": None,
            "sig_len": None,
            "duration_sec": None,
            "lead_names": [],
            "comments": [],
            "header_dx": "",
        }

    first = text[0].strip()
    parts = first.split()

    record_name = parts[0] if len(parts) >= 1 else hea_path.stem
    n_sig = safe_int(parts[1]) if len(parts) >= 2 else None
    fs = safe_float(parts[2]) if len(parts) >= 3 else None
    sig_len = safe_int(parts[3]) if len(parts) >= 4 else None
    duration_sec = None
    if fs and sig_len:
        duration_sec = sig_len / fs

    signal_lines = []
    comments = []

    for line in text[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            comments.append(s.lstrip("#").strip())
        else:
            signal_lines.append(s)

    lead_names = []
    for line in signal_lines[: n_sig or len(signal_lines)]:
        toks = line.split()
        if toks:
            lead_names.append(toks[-1])

    header_dx = ""
    for c in comments:
        if c.lower().startswith("dx:") or " dx:" in c.lower():
            header_dx = c
            break

    return {
        "status": "ok",
        "record_name": record_name,
        "n_sig": n_sig,
        "fs": fs,
        "sig_len": sig_len,
        "duration_sec": duration_sec,
        "lead_names": lead_names,
        "comments": comments,
        "header_dx": header_dx,
    }


def infer_subject_study_from_path(path):
    subject_id = ""
    study_id = ""

    for parent in [path.parent] + list(path.parents):
        name = parent.name
        if re.fullmatch(r"p\d+", name.lower()):
            subject_id = name
        if re.fullmatch(r"s\d+", name.lower()):
            study_id = name

    return subject_id, study_id


def find_matching_dat(hea_path):
    direct = hea_path.with_suffix(".dat")
    if direct.exists():
        return direct

    try:
        lines = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines[1:]:
            if line.strip().startswith("#"):
                continue
            toks = line.split()
            if toks and toks[0].endswith(".dat"):
                candidate = hea_path.parent / toks[0]
                if candidate.exists():
                    return candidate
    except Exception:
        pass

    return None


def scan_metadata_csvs(raw_dir):
    metadata = []

    for csv_path in sorted(raw_dir.rglob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                lower_cols = [c.lower() for c in cols]

                row_count = 0
                for _ in reader:
                    row_count += 1
                    if row_count >= 200000:
                        break

            likely_id_cols = [
                c for c in cols
                if c.lower() in {
                    "subject_id", "study_id", "record_id", "record_name", "path",
                    "file_name", "ecg_id", "machine_measurement_id"
                }
            ]

            likely_text_cols = [
                c for c in cols
                if any(k in c.lower() for k in [
                    "report", "interpret", "statement", "diagnosis", "comment", "text"
                ])
            ]

            metadata.append({
                "path": str(csv_path),
                "columns": cols,
                "row_count_scanned_cap_200k": row_count,
                "likely_id_columns": likely_id_cols,
                "likely_text_or_label_columns": likely_text_cols,
                "sha256": sha256_file(csv_path),
            })

        except Exception as e:
            metadata.append({
                "path": str(csv_path),
                "status": "error",
                "error": repr(e),
            })

    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/mimic_iv_ecg_demo")
    ap.add_argument("--out-dir", default="artifacts/locked_external_validation_v31")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--max-preview-records", type=int, default=20)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created = datetime.now(timezone.utc).isoformat()

    extracted = []
    if raw_dir.exists() and args.extract:
        extracted = maybe_extract_archives(raw_dir)

    hea_files = sorted(raw_dir.rglob("*.hea")) if raw_dir.exists() else []
    dat_files = sorted(raw_dir.rglob("*.dat")) if raw_dir.exists() else []
    mat_files = sorted(raw_dir.rglob("*.mat")) if raw_dir.exists() else []
    csv_files = sorted(raw_dir.rglob("*.csv")) if raw_dir.exists() else []

    rows = []
    missing_dat = []
    bad_headers = []

    for hea in hea_files:
        rel_hea = hea.relative_to(raw_dir).as_posix() if raw_dir.exists() else hea.as_posix()
        dat = find_matching_dat(hea)
        subject_id, study_id = infer_subject_study_from_path(hea)

        try:
            header = parse_hea_header(hea)
        except Exception as e:
            header = {
                "status": "header_parse_error",
                "record_name": hea.stem,
                "n_sig": None,
                "fs": None,
                "sig_len": None,
                "duration_sec": None,
                "lead_names": [],
                "comments": [],
                "header_dx": "",
                "error": repr(e),
            }
            bad_headers.append(str(hea))

        lead_names = header.get("lead_names", [])
        lead_set = set(lead_names)

        is_standard_12 = all(l in lead_set for l in STANDARD_12_LEADS)
        is_12_signal = header.get("n_sig") == 12
        is_500hz = abs((header.get("fs") or 0) - 500.0) < 1e-6
        duration = header.get("duration_sec")
        is_10s = duration is not None and 9.5 <= duration <= 10.5

        ready_for_runtime = bool(dat and is_12_signal and is_standard_12 and is_500hz and is_10s)

        if dat is None:
            missing_dat.append(str(hea))

        comments = header.get("comments", [])
        comments_preview = " | ".join(comments[:3])
        if len(comments_preview) > 500:
            comments_preview = comments_preview[:500] + "..."

        rows.append({
            "record_id": header.get("record_name") or hea.stem,
            "subject_id": subject_id,
            "study_id": study_id,
            "hea_path": str(hea),
            "dat_path": str(dat) if dat else "",
            "rel_hea_path": rel_hea,
            "n_sig": header.get("n_sig"),
            "fs": header.get("fs"),
            "sig_len": header.get("sig_len"),
            "duration_sec": duration,
            "lead_names": "|".join(lead_names),
            "is_12_signal": is_12_signal,
            "is_standard_12_lead": is_standard_12,
            "is_500hz": is_500hz,
            "is_10s": is_10s,
            "ready_for_runtime": ready_for_runtime,
            "header_dx": header.get("header_dx", ""),
            "comments_preview": comments_preview,
            "header_status": header.get("status"),
        })

    record_index = out_dir / "mimic_demo_record_index_v311.csv"
    fieldnames = [
        "record_id", "subject_id", "study_id",
        "hea_path", "dat_path", "rel_hea_path",
        "n_sig", "fs", "sig_len", "duration_sec",
        "lead_names",
        "is_12_signal", "is_standard_12_lead", "is_500hz", "is_10s",
        "ready_for_runtime",
        "header_dx", "comments_preview", "header_status",
    ]

    with record_index.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    metadata_scan = scan_metadata_csvs(raw_dir) if raw_dir.exists() else []

    ready_rows = [r for r in rows if str(r["ready_for_runtime"]) == "True"]
    paired_rows = [r for r in rows if r["dat_path"]]

    readiness = {
        "project": "CardioTwin-AI",
        "version": "v3.1.1 MIMIC-IV-ECG Demo preparation",
        "created_at_utc": created,
        "raw_dir": str(raw_dir),
        "raw_dir_exists": raw_dir.exists(),
        "extraction_requested": args.extract,
        "archives_extracted": extracted,
        "counts": {
            "hea_count": len(hea_files),
            "dat_count": len(dat_files),
            "mat_count": len(mat_files),
            "csv_count": len(csv_files),
            "record_index_rows": len(rows),
            "paired_hea_dat_rows": len(paired_rows),
            "ready_for_runtime_rows": len(ready_rows),
            "missing_dat_count": len(missing_dat),
            "bad_header_count": len(bad_headers),
        },
        "readiness": "ready_candidate" if len(ready_rows) > 0 else "dataset_not_found_or_incomplete",
        "ready_candidate_rule": "at least one WFDB .hea + .dat pair with 12 standard leads, 500 Hz, and ~10 seconds",
        "record_index_csv": str(record_index),
        "metadata_csv_scan": metadata_scan,
        "notes": [
            "This step only prepares and indexes the MIMIC-IV-ECG Demo dataset.",
            "It does not change frozen model weights, thresholds, preprocessing, or label mapping.",
            "If no files are found, download/extract MIMIC-IV-ECG Demo into data/raw/mimic_iv_ecg_demo and rerun this script.",
            "MIMIC waveform files are expected to use WFDB .hea + .dat. v3.1.2 locked dry-run evaluation should use a WFDB loader."
        ],
    }

    readiness_path = out_dir / "mimic_demo_readiness_v311.json"
    readiness_path.write_text(json.dumps(readiness, indent=2, ensure_ascii=False), encoding="utf-8")

    notes_path = out_dir / "MIMIC_IV_ECG_DEMO_PREP_NOTES_v311.md"
    notes_path.write_text(f"""# MIMIC-IV-ECG Demo Preparation Notes v3.1.1

Created: {created}

## Purpose

Prepare a local MIMIC-IV-ECG Demo folder for CardioTwin-AI v3.1 locked external validation dry-run.

## Expected Local Folder

data/raw/mimic_iv_ecg_demo

## Generated Files

- mimic_demo_readiness_v311.json
- mimic_demo_record_index_v311.csv
- MIMIC_IV_ECG_DEMO_PREP_NOTES_v311.md

## Current Readiness

{readiness["readiness"]}

## Counts

{json.dumps(readiness["counts"], indent=2, ensure_ascii=False)}

## Important Boundary

This is dataset preparation only.

Do not change frozen model weights, threshold profiles, preprocessing, label mapping, or metric definitions during locked validation.

## Next Step

If readiness is ready_candidate, proceed to v3.1.2 locked dry-run evaluation.

If readiness is dataset_not_found_or_incomplete, place or extract MIMIC-IV-ECG Demo files into data/raw/mimic_iv_ecg_demo and rerun:

python scripts/prepare_mimic_iv_ecg_demo_v311.py --extract
""", encoding="utf-8")

    # Update global v3.1 readiness report if present.
    global_readiness = out_dir / "dataset_readiness_report.json"
    if global_readiness.exists():
        try:
            obj = json.loads(global_readiness.read_text(encoding="utf-8"))
        except Exception:
            obj = {}

        obj.setdefault("dataset_candidates", {})
        obj["dataset_candidates"]["mimic_iv_ecg_demo"] = {
            "root": str(raw_dir),
            "expected_format": "WFDB .hea + .dat",
            "status": "checked_by_prepare_mimic_iv_ecg_demo_v311",
            "exists": raw_dir.exists(),
            "hea_count": len(hea_files),
            "dat_count": len(dat_files),
            "mat_count": len(mat_files),
            "csv_count": len(csv_files),
            "paired_hea_dat_rows": len(paired_rows),
            "ready_for_runtime_rows": len(ready_rows),
            "readiness": readiness["readiness"],
            "record_index_csv": str(record_index),
            "readiness_json": str(readiness_path),
        }
        obj["updated_at_utc"] = created
        obj["recommended_next_action"] = (
            "If mimic_iv_ecg_demo is ready_candidate, run v3.1.2 locked dry-run evaluation. "
            "If not, download/extract MIMIC-IV-ECG Demo into data/raw/mimic_iv_ecg_demo and rerun preparation."
        )
        global_readiness.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DONE: MIMIC-IV-ECG Demo preparation v3.1.1")
    print("RAW_DIR:", raw_dir)
    print("READINESS:", readiness_path)
    print("RECORD_INDEX:", record_index)
    print("NOTES:", notes_path)
    print("readiness:", readiness["readiness"])
    print("counts:", json.dumps(readiness["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
