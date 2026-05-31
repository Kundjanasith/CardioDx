from pathlib import Path
import argparse
import csv
import gzip
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
OUT_DEFAULT = ART / "label_supported_external_validation_v32"

ID_KEYWORDS = {
    "subject_id", "study_id", "record_id", "ecg_id", "file_name", "filename",
    "path", "cart_id", "ecg_time"
}

LABEL_KEYWORDS = [
    "diagnosis", "label", "snomed", "code", "statement", "class",
    "category", "ontology", "icd", "dx"
]

REPORT_KEYWORDS = [
    "report", "interpret", "comment", "text", "finding", "impression",
    "measurement", "machine"
]

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_text(path):
    path = Path(path)
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore", newline="")
    return path.open("r", encoding="utf-8", errors="ignore", newline="")


def find_table_files(root):
    patterns = ["*.csv", "*.csv.gz", "*.tsv", "*.tsv.gz", "*.txt", "*.txt.gz"]
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    return sorted(set(files))


def sniff_dialect(path):
    name = path.name.lower()
    if ".tsv" in name:
        return "\t"
    return ","


def scan_table(path, max_preview_rows=20):
    item = {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "columns": [],
        "n_preview_rows": 0,
        "likely_id_columns": [],
        "likely_label_columns": [],
        "likely_report_columns": [],
        "role_guess": [],
        "read_error": None,
    }

    try:
        delimiter = sniff_dialect(path)
        with open_text(path) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            cols = reader.fieldnames or []
            item["columns"] = cols

            for i, _ in enumerate(reader):
                if i >= max_preview_rows:
                    break
                item["n_preview_rows"] += 1

        for c in cols:
            lc = c.lower().strip()

            if lc in ID_KEYWORDS:
                item["likely_id_columns"].append(c)

            if any(k in lc for k in LABEL_KEYWORDS):
                item["likely_label_columns"].append(c)

            if any(k in lc for k in REPORT_KEYWORDS):
                item["likely_report_columns"].append(c)

        lname = path.name.lower()
        cols_lower = [c.lower() for c in cols]

        if "record_list" in lname or ("subject_id" in cols_lower and "study_id" in cols_lower and "path" in cols_lower):
            item["role_guess"].append("record_list_or_waveform_index")

        if "machine" in lname or any(c.startswith("report") for c in cols_lower):
            item["role_guess"].append("machine_measurements_or_report_text")

        if "snomed" in lname or "diagnosis" in lname or "label" in lname or "icd" in lname:
            item["role_guess"].append("label_or_diagnosis_metadata")

    except Exception as e:
        item["read_error"] = repr(e)

    return item


def count_files(root):
    if not root.exists():
        return {
            "hea_count": 0,
            "dat_count": 0,
            "mat_count": 0,
            "csv_count": 0,
            "csv_gz_count": 0,
            "tsv_count": 0,
            "txt_count": 0,
        }

    return {
        "hea_count": len(list(root.rglob("*.hea"))),
        "dat_count": len(list(root.rglob("*.dat"))),
        "mat_count": len(list(root.rglob("*.mat"))),
        "csv_count": len(list(root.rglob("*.csv"))),
        "csv_gz_count": len(list(root.rglob("*.csv.gz"))),
        "tsv_count": len(list(root.rglob("*.tsv"))),
        "txt_count": len(list(root.rglob("*.txt"))),
    }


def build_candidate_record_index(root, tables, out_dir, max_rows=5000):
    record_tables = [
        t for t in tables
        if "record_list_or_waveform_index" in t.get("role_guess", [])
    ]

    out_csv = out_dir / "full_mimic_candidate_record_index_v322.csv"

    if not record_tables:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["status", "message"])
            w.writerow(["not_created", "No record_list-like table found."])
        return {
            "created": False,
            "path": str(out_csv),
            "rows_written": 0,
            "reason": "No record_list-like table found.",
        }

    table_path = Path(record_tables[0]["path"])
    delimiter = sniff_dialect(table_path)

    rows_written = 0
    fieldnames_out = [
        "subject_id",
        "study_id",
        "file_name",
        "ecg_time",
        "path",
        "hea_path_guess",
        "dat_path_guess",
        "hea_exists",
        "dat_exists",
    ]

    with open_text(table_path) as f_in, out_csv.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in, delimiter=delimiter)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames_out)
        writer.writeheader()

        for row in reader:
            if rows_written >= max_rows:
                break

            rel_path = str(row.get("path", "") or row.get("file_name", "") or "").strip()
            file_name = str(row.get("file_name", "") or "").strip()

            # MIMIC record_list path usually points to WFDB base or folder path.
            base_guess = root / rel_path
            if base_guess.suffix.lower() in [".hea", ".dat"]:
                hea_guess = base_guess.with_suffix(".hea")
                dat_guess = base_guess.with_suffix(".dat")
            else:
                # If path points to a record base without extension.
                hea_guess = Path(str(base_guess) + ".hea")
                dat_guess = Path(str(base_guess) + ".dat")

                # If path is a folder and file_name is given.
                if file_name:
                    candidate = root / rel_path / file_name
                    hea_guess_alt = candidate.with_suffix(".hea")
                    dat_guess_alt = candidate.with_suffix(".dat")
                    if hea_guess_alt.exists() or dat_guess_alt.exists():
                        hea_guess = hea_guess_alt
                        dat_guess = dat_guess_alt

            writer.writerow({
                "subject_id": row.get("subject_id", ""),
                "study_id": row.get("study_id", ""),
                "file_name": file_name,
                "ecg_time": row.get("ecg_time", ""),
                "path": rel_path,
                "hea_path_guess": str(hea_guess),
                "dat_path_guess": str(dat_guess),
                "hea_exists": hea_guess.exists(),
                "dat_exists": dat_guess.exists(),
            })
            rows_written += 1

    return {
        "created": True,
        "source_table": str(table_path),
        "path": str(out_csv),
        "rows_written": rows_written,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/mimic_iv_ecg")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--max-table-scan", type=int, default=100)
    ap.add_argument("--max-index-rows", type=int, default=5000)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created = datetime.now(timezone.utc).isoformat()

    raw_dir.mkdir(parents=True, exist_ok=True)

    counts = count_files(raw_dir)

    table_files = find_table_files(raw_dir)[: args.max_table_scan]
    table_inventory = [scan_table(p) for p in table_files]

    likely_label_or_report = [
        t for t in table_inventory
        if t.get("likely_label_columns") or t.get("likely_report_columns")
    ]

    likely_record_tables = [
        t for t in table_inventory
        if "record_list_or_waveform_index" in t.get("role_guess", [])
    ]

    likely_report_tables = [
        t for t in table_inventory
        if "machine_measurements_or_report_text" in t.get("role_guess", [])
    ]

    waveform_ready = counts["hea_count"] > 0 and (counts["dat_count"] > 0 or counts["mat_count"] > 0)
    label_metadata_candidate_found = len(likely_label_or_report) > 0

    if waveform_ready and label_metadata_candidate_found:
        readiness = "ready_for_label_supported_validation_candidate"
    elif waveform_ready and not label_metadata_candidate_found:
        readiness = "waveform_ready_but_label_metadata_missing_or_unidentified"
    elif not waveform_ready and label_metadata_candidate_found:
        readiness = "label_metadata_found_but_waveforms_missing_or_unidentified"
    else:
        readiness = "not_ready"

    inventory_csv = out_dir / "full_mimic_metadata_inventory_v322.csv"
    with inventory_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "path", "name", "size_bytes", "sha256",
            "columns", "n_preview_rows",
            "likely_id_columns", "likely_label_columns", "likely_report_columns",
            "role_guess", "read_error"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in table_inventory:
            row = dict(t)
            for k in ["columns", "likely_id_columns", "likely_label_columns", "likely_report_columns", "role_guess"]:
                row[k] = "|".join(row.get(k, []))
            w.writerow(row)

    record_index_result = build_candidate_record_index(
        raw_dir,
        table_inventory,
        out_dir,
        max_rows=args.max_index_rows,
    )

    readiness_payload = {
        "project": "CardioTwin-AI",
        "version": "v3.2.2 Full MIMIC-IV-ECG access + metadata preparation",
        "created_at_utc": created,
        "raw_dir": str(raw_dir),
        "raw_dir_exists": raw_dir.exists(),
        "counts": counts,
        "waveform_ready": waveform_ready,
        "label_metadata_candidate_found": label_metadata_candidate_found,
        "readiness": readiness,
        "likely_record_tables": likely_record_tables,
        "likely_report_or_label_tables": likely_label_or_report,
        "likely_machine_report_tables": likely_report_tables,
        "metadata_inventory_csv": str(inventory_csv),
        "candidate_record_index": record_index_result,
        "download_note": {
            "physionet_access_required": True,
            "instruction": "Download Full MIMIC-IV-ECG into data/raw/mimic_iv_ecg after PhysioNet credentialed access and DUA approval.",
            "recommended_first_files": [
                "record_list.csv",
                "machine_measurements.csv or equivalent report/measurement table",
                "a small subset of WFDB waveform folders for pilot validation"
            ]
        },
        "claim_boundary": "Preparation only. No AUROC/AUPRC/F1 computed.",
    }

    readiness_json = out_dir / "full_mimic_access_readiness_v322.json"
    readiness_json.write_text(json.dumps(readiness_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    notes = out_dir / "FULL_MIMIC_IV_ECG_ACCESS_PREP_v322.md"
    notes.write_text(f"""# CardioTwin-AI v3.2.2 Full MIMIC-IV-ECG Access + Metadata Preparation

Created: {created}

## Purpose

Prepare Full MIMIC-IV-ECG for label-supported external validation.

## Local Target Folder

data/raw/mimic_iv_ecg

## Current Readiness

{readiness}

## Current Counts

{json.dumps(counts, indent=2, ensure_ascii=False)}

## What is needed

1. PhysioNet credentialed access and DUA approval.
2. Full MIMIC-IV-ECG metadata, especially record_list and report/machine-measurement fields.
3. WFDB waveform files (.hea + .dat).
4. Frozen label mapping from report/diagnosis metadata to NORM, MI, STTC, CD, HYP.

## Recommended staged download

Do not start with the full waveform download if storage/time is limited.

Recommended order:

1. Download metadata first:
   - record_list.csv
   - machine_measurements.csv or equivalent report table
2. Run this v3.2.2 preparation script again.
3. Download a small waveform subset for v3.2.3 pilot label audit.
4. Only then run larger/full validation.

## Generated Files

- full_mimic_access_readiness_v322.json
- full_mimic_metadata_inventory_v322.csv
- full_mimic_candidate_record_index_v322.csv

## Claim Boundary

Preparation only. No validation metrics are computed here.
""", encoding="utf-8")

    download_notes = out_dir / "FULL_MIMIC_IV_ECG_DOWNLOAD_NOTES_v322.md"
    download_notes.write_text("""# Full MIMIC-IV-ECG Download Notes v3.2.2

## Important

Full MIMIC-IV-ECG is credentialed PhysioNet health data. Use only your own approved PhysioNet account and comply with the Data Use Agreement.

## Folder

Place or download files under:

data/raw/mimic_iv_ecg

## Terminal download pattern

After approval, a common PhysioNet download pattern is:

wget -r -N -c -np --user YOUR_PHYSIONET_USERNAME --ask-password https://physionet.org/files/mimic-iv-ecg/1.0/

## Metadata-first strategy

For v3.2.2, metadata-first is preferred before full waveform download.

Look for:
- record_list.csv
- machine_measurements.csv
- report/diagnosis/statement tables if provided
- WFDB folders with .hea + .dat

## After download

Run:

python scripts/prepare_full_mimic_iv_ecg_access_metadata_v322.py

Then rerun:

python scripts/probe_v32_label_supported_datasets.py
""", encoding="utf-8")

    # Update v3.2 dataset plan if present
    plan_path = out_dir / "dataset_selection_plan_v32.json"
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            plan = {}
        plan.setdefault("v3_2_2_full_mimic_preparation", {})
        plan["v3_2_2_full_mimic_preparation"] = {
            "updated_at_utc": created,
            "readiness": readiness,
            "counts": counts,
            "readiness_json": str(readiness_json),
            "metadata_inventory_csv": str(inventory_csv),
            "candidate_record_index": record_index_result,
        }
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DONE: v3.2.2 Full MIMIC-IV-ECG access + metadata preparation")
    print("RAW_DIR:", raw_dir)
    print("READINESS_JSON:", readiness_json)
    print("INVENTORY_CSV:", inventory_csv)
    print("NOTES:", notes)
    print("DOWNLOAD_NOTES:", download_notes)
    print("readiness:", readiness)
    print("counts:", json.dumps(counts, indent=2, ensure_ascii=False))
    print("candidate_record_index:", json.dumps(record_index_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
