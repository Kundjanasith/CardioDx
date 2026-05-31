from pathlib import Path
import argparse
import csv
import json
import re
import zipfile
import hashlib
from datetime import datetime, timezone
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

ROOT = Path(".")
ART = ROOT / "artifacts"
OUT = ART / "label_supported_external_validation_v32"
RELEASE = ART / "release_rc1"
OUT.mkdir(parents=True, exist_ok=True)
RELEASE.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
REPORT_COLS = [f"report_{i}" for i in range(18)]

# v3.2.4 conservative frozen weak-label mapping.
# Intentionally avoids overly broad phrases such as generic "MI" and atrial enlargement.
MAPPING_RULES = {
    "NORM": {
        "include": [
            r"\bnormal ecg\b",
            r"\bnormal sinus rhythm\b",
            r"\bwithin normal limits\b",
            r"\botherwise normal\b",
            r"\bnormal tracing\b",
        ],
        "exclude_if_any_abnormal_label": True,
        "notes": "NORM is assigned only when a normal phrase is found and no abnormal superclass rule is positive."
    },
    "MI": {
        "include": [
            r"\bmyocardial infarction\b",
            r"\bold myocardial infarction\b",
            r"\bacute myocardial infarction\b",
            r"\binferior infarct(?:ion)?\b",
            r"\banterior infarct(?:ion)?\b",
            r"\bseptal infarct(?:ion)?\b",
            r"\blateral infarct(?:ion)?\b",
            r"\bposterior infarct(?:ion)?\b",
            r"\bold infarct(?:ion)?\b",
            r"\bacute infarct(?:ion)?\b",
        ],
        "exclude_near_left": [
            r"\bno evidence of\b",
            r"\bwithout evidence of\b",
            r"\bcannot rule out\b",
            r"\brule out\b",
            r"\br/o\b",
            r"\bpossible\b",
            r"\bprobably\b",
            r"\bprobable\b",
            r"\bquestion(?:able)?\b",
            r"\bsuspect(?:ed)?\b",
        ],
        "notes": "Generic word-boundary MI is intentionally excluded; uncertain infarct phrases are excluded."
    },
    "STTC": {
        "include": [
            r"\bst[- ]?t abnormal(?:ity|ities)?\b",
            r"\bst depression\b",
            r"\bst elevation\b",
            r"\bt wave abnormal(?:ity|ities)?\b",
            r"\bt-wave abnormal(?:ity|ities)?\b",
            r"\bischemia\b",
            r"\bischaemia\b",
            r"\brepolarization abnormal(?:ity|ities)?\b",
            r"\brepolarisation abnormal(?:ity|ities)?\b",
        ],
        "exclude_near_left": [
            r"\bno evidence of\b",
            r"\bwithout evidence of\b",
            r"\bcannot rule out\b",
            r"\brule out\b",
            r"\br/o\b",
            r"\bpossible\b",
            r"\bquestion(?:able)?\b",
        ],
        "notes": "STTC mapping uses explicit ST-T/T-wave/ischemia/repolarization phrases."
    },
    "CD": {
        "include": [
            r"\bright bundle branch block\b",
            r"\bleft bundle branch block\b",
            r"\brbbb\b",
            r"\blbbb\b",
            r"\bav block\b",
            r"\bfirst degree av block\b",
            r"\b1st degree av block\b",
            r"\bintraventricular conduction delay\b",
            r"\bconduction delay\b",
            r"\bfascicular block\b",
            r"\bhemiblock\b",
        ],
        "exclude_near_left": [
            r"\bno evidence of\b",
            r"\bwithout evidence of\b",
            r"\bcannot rule out\b",
            r"\brule out\b",
            r"\br/o\b",
            r"\bpossible\b",
            r"\bquestion(?:able)?\b",
        ],
        "notes": "CD mapping uses conduction/block phrases only."
    },
    "HYP": {
        "include": [
            r"\bleft ventricular hypertrophy\b",
            r"\bright ventricular hypertrophy\b",
            r"\blvh\b",
            r"\brvh\b",
            r"\bventricular hypertrophy\b",
        ],
        "exclude_near_left": [
            r"\bno evidence of\b",
            r"\bwithout evidence of\b",
            r"\bcannot rule out\b",
            r"\brule out\b",
            r"\br/o\b",
            r"\bpossible\b",
            r"\bquestion(?:able)?\b",
        ],
        "notes": "Atrial enlargement is intentionally excluded from HYP in v3.2.4 conservative mapping."
    }
}


def normalize_text(x):
    if pd.isna(x):
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def report_text_from_row(row):
    parts = []
    for c in REPORT_COLS:
        if c in row:
            v = normalize_text(row[c])
            if v:
                parts.append(v)
    return " | ".join(parts)


def has_near_left_exclusion(text_lower, start, patterns, window=55):
    left = text_lower[max(0, start - window):start]
    for pat in patterns:
        if re.search(pat, left, flags=re.IGNORECASE):
            return True, pat
    return False, ""


def match_label(text, label):
    t = text.lower()
    rule = MAPPING_RULES[label]
    evidence = []

    for pat in rule.get("include", []):
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            excluded, exclude_pat = has_near_left_exclusion(
                t,
                m.start(),
                rule.get("exclude_near_left", []),
            )
            span0 = max(0, m.start() - 60)
            span1 = min(len(text), m.end() + 100)
            snippet = text[span0:span1].strip()
            evidence.append({
                "pattern": pat,
                "match": m.group(0),
                "excluded_by_context": excluded,
                "exclude_pattern": exclude_pat,
                "snippet": snippet,
            })

    safe_evidence = [e for e in evidence if not e["excluded_by_context"]]
    return len(safe_evidence) > 0, safe_evidence, evidence


def load_record_lookup(record_list_csv, root):
    df = pd.read_csv(record_list_csv, low_memory=False)
    lookup = {}

    for _, r in df.iterrows():
        sid = str(r.get("subject_id", "")).strip()
        stid = str(r.get("study_id", "")).strip()
        path = str(r.get("path", "")).strip()
        file_name = str(r.get("file_name", "")).strip()
        ecg_time = str(r.get("ecg_time", "")).strip()

        key = (sid, stid)
        base = root / path

        hea_guess = Path(str(base) + ".hea")
        dat_guess = Path(str(base) + ".dat")

        lookup[key] = {
            "file_name": file_name,
            "ecg_time_record_list": ecg_time,
            "path": path,
            "hea_path_guess": str(hea_guess),
            "dat_path_guess": str(dat_guess),
            "hea_exists": hea_guess.exists(),
            "dat_exists": dat_guess.exists(),
        }

    return lookup


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine-csv", default="data/raw/mimic_iv_ecg/machine_measurements.csv")
    ap.add_argument("--record-list", default="data/raw/mimic_iv_ecg/record_list.csv")
    ap.add_argument("--raw-root", default="data/raw/mimic_iv_ecg")
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = full metadata")
    ap.add_argument("--sample-per-class", type=int, default=80)
    ap.add_argument("--subset-per-class", type=int, default=120)
    args = ap.parse_args()

    machine_csv = Path(args.machine_csv)
    record_list_csv = Path(args.record_list)
    raw_root = Path(args.raw_root)

    if not machine_csv.exists():
        raise FileNotFoundError(machine_csv)
    if not record_list_csv.exists():
        raise FileNotFoundError(record_list_csv)

    created = datetime.now(timezone.utc).isoformat()

    print("[INFO] loading record_list lookup...")
    record_lookup = load_record_lookup(record_list_csv, raw_root)
    print("[INFO] record_list rows:", len(record_lookup))

    preview = pd.read_csv(machine_csv, nrows=1)
    available_cols = list(preview.columns)
    id_cols = [c for c in ["subject_id", "study_id", "cart_id", "ecg_time"] if c in available_cols]
    report_cols = [c for c in REPORT_COLS if c in available_cols]
    usecols = id_cols + report_cols

    label_index_path = OUT / "full_mimic_frozen_label_index_v324.csv"
    examples_path = OUT / "full_mimic_frozen_label_examples_v324.csv"
    support_csv = OUT / "full_mimic_frozen_label_support_v324.csv"
    subset_plan_path = OUT / "full_mimic_waveform_subset_plan_v324.csv"

    total = 0
    rows_with_report = 0
    mapped_any = 0
    support = Counter()
    excluded_context_counts = Counter()
    cooccur = defaultdict(Counter)
    examples = []
    subset_rows = []
    subset_seen_keys = set()

    with label_index_path.open("w", encoding="utf-8", newline="") as f_index:
        fieldnames = [
            "subject_id", "study_id", "cart_id", "ecg_time",
            "file_name", "path",
            "hea_path_guess", "dat_path_guess", "hea_exists", "dat_exists",
            "label_NORM", "label_MI", "label_STTC", "label_CD", "label_HYP",
            "positive_labels",
            "mapping_evidence",
            "report_text_preview",
        ]
        w_index = csv.DictWriter(f_index, fieldnames=fieldnames)
        w_index.writeheader()

        for chunk in pd.read_csv(machine_csv, usecols=usecols, chunksize=args.chunksize, low_memory=False):
            for _, row in chunk.iterrows():
                if args.max_rows and total >= args.max_rows:
                    break

                total += 1

                sid = str(row.get("subject_id", "")).strip()
                stid = str(row.get("study_id", "")).strip()
                cart_id = str(row.get("cart_id", "")).strip()
                ecg_time = str(row.get("ecg_time", "")).strip()

                text = report_text_from_row(row)
                if text:
                    rows_with_report += 1

                raw_label_results = {}
                full_evidence = {}
                excluded_by_label = {}

                for label in ["MI", "STTC", "CD", "HYP"]:
                    ok, safe_ev, all_ev = match_label(text, label)
                    raw_label_results[label] = int(ok)
                    full_evidence[label] = safe_ev[:3]
                    excluded_by_label[label] = [e for e in all_ev if e.get("excluded_by_context")]

                    if excluded_by_label[label]:
                        excluded_context_counts[label] += len(excluded_by_label[label])

                # NORM only when no abnormal label is present.
                norm_ok, norm_safe_ev, norm_all_ev = match_label(text, "NORM")
                raw_label_results["NORM"] = int(norm_ok and sum(raw_label_results[l] for l in ["MI", "STTC", "CD", "HYP"]) == 0)
                full_evidence["NORM"] = norm_safe_ev[:3] if raw_label_results["NORM"] else []

                positive = [label for label in TARGET_CLASSES if raw_label_results.get(label, 0) == 1]

                if positive:
                    mapped_any += 1

                for label in positive:
                    support[label] += 1

                for a in positive:
                    for b in positive:
                        cooccur[a][b] += 1

                lookup = record_lookup.get((sid, stid), {
                    "file_name": "",
                    "ecg_time_record_list": "",
                    "path": "",
                    "hea_path_guess": "",
                    "dat_path_guess": "",
                    "hea_exists": False,
                    "dat_exists": False,
                })

                evidence_short = []
                for label in positive:
                    evs = full_evidence.get(label, [])
                    if evs:
                        evidence_short.append(f"{label}:{evs[0].get('match', '')}")

                out_row = {
                    "subject_id": sid,
                    "study_id": stid,
                    "cart_id": cart_id,
                    "ecg_time": ecg_time,
                    "file_name": lookup.get("file_name", ""),
                    "path": lookup.get("path", ""),
                    "hea_path_guess": lookup.get("hea_path_guess", ""),
                    "dat_path_guess": lookup.get("dat_path_guess", ""),
                    "hea_exists": lookup.get("hea_exists", False),
                    "dat_exists": lookup.get("dat_exists", False),
                    "label_NORM": raw_label_results["NORM"],
                    "label_MI": raw_label_results["MI"],
                    "label_STTC": raw_label_results["STTC"],
                    "label_CD": raw_label_results["CD"],
                    "label_HYP": raw_label_results["HYP"],
                    "positive_labels": "|".join(positive),
                    "mapping_evidence": " ; ".join(evidence_short),
                    "report_text_preview": text[:500],
                }
                w_index.writerow(out_row)

                # Examples
                for label in positive:
                    if sum(1 for e in examples if e["label"] == label) < args.sample_per_class:
                        evs = full_evidence.get(label, [])
                        examples.append({
                            "label": label,
                            "subject_id": sid,
                            "study_id": stid,
                            "cart_id": cart_id,
                            "path": lookup.get("path", ""),
                            "match": evs[0].get("match", "") if evs else "",
                            "snippet": evs[0].get("snippet", "")[:500] if evs else "",
                            "positive_labels": "|".join(positive),
                            "report_text_preview": text[:500],
                        })

                # Subset plan
                for label in positive:
                    if sum(1 for r in subset_rows if r["target_label"] == label) < args.subset_per_class:
                        key = (sid, stid, label)
                        if key not in subset_seen_keys:
                            subset_seen_keys.add(key)
                            subset_rows.append({
                                "target_label": label,
                                "subject_id": sid,
                                "study_id": stid,
                                "cart_id": cart_id,
                                "positive_labels": "|".join(positive),
                                "path": lookup.get("path", ""),
                                "hea_path_guess": lookup.get("hea_path_guess", ""),
                                "dat_path_guess": lookup.get("dat_path_guess", ""),
                                "hea_exists": lookup.get("hea_exists", False),
                                "dat_exists": lookup.get("dat_exists", False),
                                "mapping_evidence": " ; ".join(evidence_short),
                            })

            if args.max_rows and total >= args.max_rows:
                break

    # Write examples
    with examples_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "label", "subject_id", "study_id", "cart_id", "path",
            "match", "snippet", "positive_labels", "report_text_preview"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(examples)

    # Write subset plan
    with subset_plan_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "target_label", "subject_id", "study_id", "cart_id",
            "positive_labels", "path", "hea_path_guess", "dat_path_guess",
            "hea_exists", "dat_exists", "mapping_evidence"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(subset_rows)

    support_rows = []
    for label in TARGET_CLASSES:
        support_rows.append({
            "target_superclass": label,
            "frozen_positive_count": int(support[label]),
            "frozen_positive_pct": float(support[label] / total) if total else 0.0,
            "excluded_context_match_count": int(excluded_context_counts[label]),
        })

    with support_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "target_superclass",
            "frozen_positive_count",
            "frozen_positive_pct",
            "excluded_context_match_count",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(support_rows)

    mapping_json = OUT / "frozen_report_to_superclass_mapping_v324.json"
    mapping_payload = {
        "project": "CardioTwin-AI",
        "version": "v3.2.4 conservative frozen report-to-superclass weak-label mapping",
        "created_at_utc": created,
        "target_classes": TARGET_CLASSES,
        "mapping_rules": MAPPING_RULES,
        "important_design_decisions": [
            "Generic standalone MI is excluded from MI mapping.",
            "Cannot-rule-out / possible / suspected / question contexts are excluded for abnormal labels.",
            "Atrial enlargement is excluded from HYP in this conservative version.",
            "NORM is only assigned when a normal phrase is present and no abnormal superclass is positive.",
        ],
        "claim_boundary": "Frozen weak-label mapping for research validation preparation only. Not physician-adjudicated diagnosis."
    }
    mapping_json.write_text(json.dumps(mapping_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_json = OUT / "full_mimic_frozen_label_support_v324.json"
    summary = {
        "project": "CardioTwin-AI",
        "version": "v3.2.4 Full MIMIC-IV-ECG conservative frozen label mapping",
        "created_at_utc": created,
        "total_rows_scanned": int(total),
        "rows_with_report_text": int(rows_with_report),
        "mapped_any_label_count": int(mapped_any),
        "mapped_any_label_pct": float(mapped_any / total) if total else 0.0,
        "support": {
            r["target_superclass"]: {
                "frozen_positive_count": r["frozen_positive_count"],
                "frozen_positive_pct": r["frozen_positive_pct"],
                "excluded_context_match_count": r["excluded_context_match_count"],
            }
            for r in support_rows
        },
        "cooccurrence_matrix": {
            a: {b: int(cooccur[a][b]) for b in TARGET_CLASSES}
            for a in TARGET_CLASSES
        },
        "outputs": {
            "mapping_json": str(mapping_json),
            "label_index_csv": str(label_index_path),
            "support_csv": str(support_csv),
            "examples_csv": str(examples_path),
            "waveform_subset_plan_csv": str(subset_plan_path),
        },
        "next_step": "Download the waveform paths listed in full_mimic_waveform_subset_plan_v324.csv, then run v3.2.5 waveform subset readiness.",
        "claim_boundary": "Label mapping preparation only. No model validation metrics computed."
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = OUT / "FULL_MIMIC_FROZEN_LABEL_MAPPING_v324.md"
    md.write_text(f"""# CardioTwin-AI v3.2.4 Conservative Frozen Label Mapping

Created: {created}

## Purpose

Freeze a conservative report-to-superclass weak-label mapping for Full MIMIC-IV-ECG.

## Why Conservative Mapping Is Needed

The v3.2.3 audit found large candidate support, but MI had many uncertain or negated contexts.

Therefore v3.2.4 intentionally avoids broad rules and excludes phrases such as:

- cannot rule out
- possible
- probable
- question/questionable
- suspected
- no evidence of

## Rows

- Total rows scanned: {total}
- Rows with report text: {rows_with_report}
- Rows mapped to at least one label: {mapped_any}

## Frozen Label Support

{json.dumps(summary["support"], indent=2, ensure_ascii=False)}

## Important Design Decisions

{json.dumps(mapping_payload["important_design_decisions"], indent=2, ensure_ascii=False)}

## Outputs

- `{mapping_json}`
- `{label_index_path}`
- `{support_csv}`
- `{examples_path}`
- `{subset_plan_path}`

## Next Step

Proceed to v3.2.5:

Download the waveform records listed in:

`{subset_plan_path}`

Then run waveform subset readiness and locked pilot validation.

## Claim Boundary

This is weak-label mapping preparation only. No AUROC/AUPRC/F1 is computed here.
""", encoding="utf-8")

    html = OUT / "full_mimic_frozen_label_mapping_v324.html"
    html.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CardioTwin-AI v3.2.4 Frozen Label Mapping</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>CardioTwin-AI v3.2.4 Conservative Frozen Label Mapping</h1>
  <div class="warning">
    Weak-label mapping preparation only. No AUROC/AUPRC/F1 computed.
  </div>
  <h2>Summary</h2>
  <pre>{json.dumps(summary, indent=2, ensure_ascii=False)}</pre>
</body>
</html>
""", encoding="utf-8")

    # Package
    zip_path = RELEASE / "cardiotwin_v3_2_4_full_mimic_frozen_label_mapping_pack.zip"
    manifest_path = RELEASE / "cardiotwin_v3_2_4_full_mimic_frozen_label_mapping_manifest.json"

    files = [
        mapping_json,
        summary_json,
        support_csv,
        examples_path,
        subset_plan_path,
        md,
        html,
        OUT / "full_mimic_report_field_audit_v323.json",
        OUT / "full_mimic_report_keyword_support_v323.csv",
        OUT / "full_mimic_report_label_candidates_v323.csv",
    ]
    files = [p for p in files if p.exists()]

    manifest = {
        "project": "CardioTwin-AI",
        "version": "v3.2.4 Full MIMIC-IV-ECG Frozen Label Mapping Pack",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "claim_boundary": "Weak-label mapping preparation only. No model validation metrics computed.",
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
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.as_posix())
        z.write(manifest_path, manifest_path.as_posix())

    # Update dataset plan
    plan_path = OUT / "dataset_selection_plan_v32.json"
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            plan = {}
        plan["v3_2_4_frozen_label_mapping"] = {
            "updated_at_utc": created,
            "status": "completed",
            "summary_json": str(summary_json),
            "mapping_json": str(mapping_json),
            "label_index_csv": str(label_index_path),
            "support": summary["support"],
            "subset_plan_csv": str(subset_plan_path),
        }
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DONE: v3.2.4 conservative frozen label mapping")
    print("MAPPING_JSON:", mapping_json)
    print("SUMMARY_JSON:", summary_json)
    print("LABEL_INDEX:", label_index_path)
    print("SUPPORT_CSV:", support_csv)
    print("EXAMPLES:", examples_path)
    print("SUBSET_PLAN:", subset_plan_path)
    print("ZIP:", zip_path)
    print("MANIFEST:", manifest_path)
    print(json.dumps({
        "total_rows_scanned": total,
        "rows_with_report_text": rows_with_report,
        "mapped_any_label_count": mapped_any,
        "support": summary["support"],
        "files_indexed": manifest["files_indexed"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
