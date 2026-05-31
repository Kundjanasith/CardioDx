from pathlib import Path
import argparse
import csv
import json
import re
import hashlib
import zipfile
from datetime import datetime, timezone
from collections import defaultdict, Counter

ROOT = Path(".")
V32 = Path("artifacts/public_multicenter_validation_v32")
OUT = Path("artifacts/public_multicenter_validation_v33")
RELEASE = Path("artifacts/release_rc1")
OUT.mkdir(parents=True, exist_ok=True)
RELEASE.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ABNORMAL_CLASSES = ["MI", "STTC", "CD", "HYP"]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text_safe(path, max_bytes=100000):
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def split_labels(x):
    if x is None:
        return []
    return [p.strip() for p in re.split(r"[|,;/ ]+", str(x)) if p.strip()]


def load_code_map():
    code_to_labels = defaultdict(set)
    mapping_sources = []

    # Project-specific harmonization maps first.
    config_candidates = [
        Path("configs/cinc2020_to_ptbxl_superclass_map_v21.csv"),
        Path("configs/cinc2020_to_ptbxl_superclass_map.csv"),
    ]

    for cfg in config_candidates:
        if not cfg.exists():
            continue

        with cfg.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            lower = {c.lower().strip(): c for c in cols}

            code_col = lower.get("code") or lower.get("snomed_code") or lower.get("snomed ct code")
            cls_col = lower.get("ptbxl_superclass") or lower.get("target_superclass") or lower.get("superclass")
            decision_col = lower.get("decision")

            if not code_col or not cls_col:
                continue

            for row in reader:
                code = str(row.get(code_col, "")).strip()
                cls = str(row.get(cls_col, "")).strip()
                decision = "include"
                if decision_col:
                    decision = str(row.get(decision_col, "")).strip().lower()

                if code and cls in TARGET_CLASSES and decision == "include":
                    code_to_labels[code].add(cls)

        mapping_sources.append(str(cfg))

    # Public SNOMED mapping created in v3.2-public.1.
    public_maps = [
        V32 / "dx_mapping_public_superclass_v321.csv",
        V32 / "public_snomed_to_superclass_mapping_v321.csv",
    ]

    for mp in public_maps:
        if not mp.exists():
            continue

        with mp.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            lower = {c.lower().strip(): c for c in cols}

            code_col = (
                lower.get("snomed ct code")
                or lower.get("snomed_code")
                or lower.get("snomed code")
                or lower.get("code")
            )
            target_col = lower.get("target_superclasses") or lower.get("target_superclass")

            if not code_col or not target_col:
                continue

            for row in reader:
                code = str(row.get(code_col, "")).strip()
                if not code:
                    continue

                for lab in split_labels(row.get(target_col, "")):
                    if lab in TARGET_CLASSES:
                        code_to_labels[code].add(lab)

        mapping_sources.append(str(mp))

    return {k: sorted(v) for k, v in code_to_labels.items()}, mapping_sources


def parse_hea(path):
    text = read_text_safe(path)
    lines = text.splitlines()
    first = lines[0].split() if lines else []

    n_sig = None
    fs = None

    try:
        if len(first) >= 2:
            n_sig = int(first[1])
        if len(first) >= 3:
            fs = float(first[2].split("/")[0])
    except Exception:
        pass

    dx_tokens = []
    for line in lines:
        s = line.strip()
        if re.match(r"^#\s*Dx\s*:", s, flags=re.IGNORECASE):
            rhs = s.split(":", 1)[1]
            dx_tokens = [x.strip() for x in re.split(r"[,;|]", rhs) if x.strip()]
            break

    return {
        "record_id": path.stem,
        "hea_path": str(path),
        "n_sig": n_sig,
        "fs": fs,
        "dx_tokens": dx_tokens,
    }


def map_tokens(tokens, code_to_labels):
    labels = set()
    unmapped = []

    for tok in tokens:
        raw = str(tok).strip()
        digits = re.sub(r"\D", "", raw)

        if digits and digits in code_to_labels:
            for lab in code_to_labels[digits]:
                labels.add(lab)
        else:
            unmapped.append(raw)

    return sorted(labels), unmapped


def metric_labels_from_mapped(mapped):
    mapped = set(mapped)
    out = set()

    for lab in ABNORMAL_CLASSES:
        if lab in mapped:
            out.add(lab)

    # Claim-safe NORM: only positive if no abnormal target class is present.
    if "NORM" in mapped and not out:
        out.add("NORM")

    return sorted(out)


def load_registry_ready_sources():
    reg_path = V32 / "public_dataset_registry_v32.json"
    if not reg_path.exists():
        raise FileNotFoundError(reg_path)

    obj = json.loads(reg_path.read_text(encoding="utf-8"))
    sources = []

    for s in obj.get("sources", []):
        if s.get("readiness") == "ready_for_public_locked_validation_candidate":
            roots = [Path(x) for x in s.get("existing_roots", []) if Path(x).exists()]
            sources.append({
                "source_id": s.get("source_id"),
                "role": s.get("role"),
                "claim_role": s.get("claim_role"),
                "roots": roots,
            })

    return sources, obj


def candidate_signal_paths(hea_path):
    base = hea_path.with_suffix("")
    mat = Path(str(base) + ".mat")
    dat = Path(str(base) + ".dat")
    return {
        "mat_path": str(mat),
        "dat_path": str(dat),
        "mat_exists": mat.exists(),
        "dat_exists": dat.exists(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source-label", type=int, default=300)
    ap.add_argument("--max-total-per-source", type=int, default=2500)
    ap.add_argument("--include-all-ready", action="store_true")
    args = ap.parse_args()

    created = datetime.now(timezone.utc).isoformat()

    code_to_labels, mapping_sources = load_code_map()
    sources, registry = load_registry_ready_sources()

    rows = []
    source_label_counts = defaultdict(Counter)
    source_total_counts = Counter()
    unmapped_counts = defaultdict(Counter)
    scanned_counts = Counter()

    for src in sources:
        source_id = src["source_id"]
        hea_files = []

        for root in src["roots"]:
            hea_files.extend(sorted(root.rglob("*.hea")))

        selected_records = set()

        for hp in hea_files:
            scanned_counts[source_id] += 1

            if not args.include_all_ready and source_total_counts[source_id] >= args.max_total_per_source:
                break

            h = parse_hea(hp)

            if h["n_sig"] != 12:
                continue

            mapped, unmapped = map_tokens(h["dx_tokens"], code_to_labels)
            metric_labels = metric_labels_from_mapped(mapped)

            for u in unmapped:
                unmapped_counts[source_id][u] += 1

            if not metric_labels:
                continue

            if args.include_all_ready:
                keep = True
            else:
                keep = False
                for lab in metric_labels:
                    if source_label_counts[source_id][lab] < args.per_source_label:
                        keep = True
                        break

            if not keep:
                continue

            key = (source_id, h["record_id"])
            if key in selected_records:
                continue
            selected_records.add(key)

            sig = candidate_signal_paths(hp)

            out = {
                "source_id": source_id,
                "role": src["role"],
                "claim_role": src["claim_role"],
                "record_id": h["record_id"],
                "hea_path": h["hea_path"],
                "mat_path": sig["mat_path"],
                "dat_path": sig["dat_path"],
                "mat_exists": sig["mat_exists"],
                "dat_exists": sig["dat_exists"],
                "waveform_ready": bool(sig["mat_exists"] or sig["dat_exists"]),
                "n_sig": h["n_sig"],
                "fs": h["fs"],
                "dx_tokens": "|".join(h["dx_tokens"]),
                "mapped_labels_raw": "|".join(mapped),
                "metric_labels": "|".join(metric_labels),
                "label_NORM": int("NORM" in metric_labels),
                "label_MI": int("MI" in metric_labels),
                "label_STTC": int("STTC" in metric_labels),
                "label_CD": int("CD" in metric_labels),
                "label_HYP": int("HYP" in metric_labels),
                "locked_split": "public_external_test",
            }

            rows.append(out)
            source_total_counts[source_id] += 1

            for lab in metric_labels:
                source_label_counts[source_id][lab] += 1

    cohort_csv = OUT / "public_locked_validation_cohort_v330.csv"
    summary_json = OUT / "public_locked_validation_cohort_summary_v330.json"
    protocol_md = OUT / "PUBLIC_LOCKED_VALIDATION_COHORT_PROTOCOL_v330.md"
    unmapped_csv = OUT / "public_locked_validation_unmapped_dx_v330.csv"

    fieldnames = [
        "source_id", "role", "claim_role", "record_id",
        "hea_path", "mat_path", "dat_path", "mat_exists", "dat_exists", "waveform_ready",
        "n_sig", "fs", "dx_tokens", "mapped_labels_raw", "metric_labels",
        "label_NORM", "label_MI", "label_STTC", "label_CD", "label_HYP",
        "locked_split",
    ]

    with cohort_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    unmapped_rows = []
    for source_id, counter in unmapped_counts.items():
        for dx, n in counter.most_common(100):
            unmapped_rows.append({
                "source_id": source_id,
                "dx_token": dx,
                "count": int(n),
            })

    with unmapped_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "dx_token", "count"])
        w.writeheader()
        w.writerows(unmapped_rows)

    label_totals = Counter()
    source_table = {}

    for r in rows:
        sid = r["source_id"]
        if sid not in source_table:
            source_table[sid] = {
                "records": 0,
                "NORM": 0,
                "MI": 0,
                "STTC": 0,
                "CD": 0,
                "HYP": 0,
            }

        source_table[sid]["records"] += 1

        for lab in TARGET_CLASSES:
            if int(r[f"label_{lab}"]) == 1:
                source_table[sid][lab] += 1
                label_totals[lab] += 1

    summary = {
        "project": "CardioTwin-AI",
        "version": "v3.3.0 public locked validation cohort",
        "created_at_utc": created,
        "purpose": "Freeze source-separated public ECG validation cohort before locked model inference.",
        "cohort_csv": str(cohort_csv),
        "total_records_selected": len(rows),
        "source_table": source_table,
        "label_totals": {lab: int(label_totals[lab]) for lab in TARGET_CLASSES},
        "scanned_counts": {k: int(v) for k, v in scanned_counts.items()},
        "per_source_label_cap": args.per_source_label,
        "max_total_per_source": args.max_total_per_source,
        "include_all_ready": bool(args.include_all_ready),
        "mapping_sources": mapping_sources,
        "code_to_superclass_size": len(code_to_labels),
        "norm_policy": "NORM is counted as positive only when no abnormal target superclass is present.",
        "source_policy": "Metrics must be reported by source; do not pool as a random split.",
        "next_step": "v3.3.1 run frozen CardioTwin-AI v3.0.4.1 inference on this locked cohort.",
        "claim_boundary": "Locked public validation cohort only. No model performance metrics computed in this step.",
        "outputs": {
            "cohort_csv": str(cohort_csv),
            "summary_json": str(summary_json),
            "protocol_md": str(protocol_md),
            "unmapped_dx_csv": str(unmapped_csv),
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    protocol_md.write_text(
        "# CardioTwin-AI v3.3.0 Public Locked Validation Cohort\n\n"
        f"Created: {created}\n\n"
        "## Purpose\n\n"
        "Freeze a source-separated public ECG validation cohort before locked inference.\n\n"
        "## Frozen Rules\n\n"
        "- Use only sources marked ready_for_public_locked_validation_candidate in v3.2-public.\n"
        "- Preserve source_id for every record.\n"
        "- Do not pool all records into a random split.\n"
        "- Use frozen NORM policy: NORM is positive only when no abnormal target superclass is present.\n"
        "- Keep multi-label abnormal targets as multi-label positives.\n\n"
        "## Outputs\n\n"
        f"- {cohort_csv}\n"
        f"- {summary_json}\n"
        f"- {unmapped_csv}\n\n"
        "## Summary\n\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n\n## Claim Boundary\n\n"
        "This step freezes the cohort only. No AUROC, AUPRC, F1, or sensitivity is computed here.\n",
        encoding="utf-8",
    )

    zip_path = RELEASE / "cardiotwin_v3_3_0_public_locked_validation_cohort_pack.zip"
    manifest_path = RELEASE / "cardiotwin_v3_3_0_public_locked_validation_cohort_manifest.json"

    files = [cohort_csv, summary_json, protocol_md, unmapped_csv]
    manifest = {
        "project": "CardioTwin-AI",
        "version": "v3.3.0 Public Locked Validation Cohort Pack",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_indexed": len(files),
        "summary": summary,
        "files": [
            {
                "path": p.as_posix(),
                "size_bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            }
            for p in files
        ],
        "claim_boundary": summary["claim_boundary"],
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.as_posix())
        z.write(manifest_path, manifest_path.as_posix())

    print("DONE: v3.3.0 public locked validation cohort")
    print("COHORT:", cohort_csv)
    print("SUMMARY:", summary_json)
    print("PROTOCOL:", protocol_md)
    print("UNMAPPED:", unmapped_csv)
    print("ZIP:", zip_path)
    print("MANIFEST:", manifest_path)
    print(json.dumps({
        "total_records_selected": len(rows),
        "source_table": source_table,
        "label_totals": summary["label_totals"],
        "code_to_superclass_size": len(code_to_labels),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
