from pathlib import Path
import argparse
import json
import re
import csv
from datetime import datetime, timezone
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

ROOT = Path(".")
ART = ROOT / "artifacts"
OUT = ART / "label_supported_external_validation_v32"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
REPORT_COLS = [f"report_{i}" for i in range(18)]

# Conservative keyword rules.
# These are candidates only. They are NOT frozen mappings yet.
RULES = {
    "NORM": [
        r"\bnormal ecg\b",
        r"\bnormal sinus rhythm\b",
        r"\botherwise normal\b",
        r"\bwithin normal limits\b",
        r"\bnormal tracing\b",
    ],
    "MI": [
        r"\bmyocardial infarction\b",
        r"\bacute mi\b",
        r"\bold mi\b",
        r"\binferior infarct\b",
        r"\banterior infarct\b",
        r"\bseptal infarct\b",
        r"\blateral infarct\b",
        r"\bposterior infarct\b",
        r"\binfarct\b",
    ],
    "STTC": [
        r"\bst[- ]?t abnormal",
        r"\bst depression\b",
        r"\bst elevation\b",
        r"\bt wave abnormal",
        r"\bt-wave abnormal",
        r"\bischemia\b",
        r"\bischaemia\b",
        r"\bnonspecific st",
        r"\brepolarization abnormal",
        r"\brepolarisation abnormal",
    ],
    "CD": [
        r"\bright bundle branch block\b",
        r"\bleft bundle branch block\b",
        r"\brbbb\b",
        r"\blbbb\b",
        r"\bav block\b",
        r"\b1st degree av block\b",
        r"\bfirst degree av block\b",
        r"\bintraventricular conduction delay\b",
        r"\bconduction delay\b",
        r"\bfascicular block\b",
        r"\bhemiblock\b",
    ],
    "HYP": [
        r"\bleft ventricular hypertrophy\b",
        r"\bright ventricular hypertrophy\b",
        r"\blvh\b",
        r"\brvh\b",
        r"\bventricular hypertrophy\b",
        r"\batrial enlargement\b",
        r"\bleft atrial enlargement\b",
        r"\bright atrial enlargement\b",
    ],
}

NEGATION_PATTERNS = [
    r"\bno evidence of\b",
    r"\bwithout evidence of\b",
    r"\brule out\b",
    r"\br/o\b",
    r"\bquestion\b",
    r"\bpossible\b",
    r"\bcannot rule out\b",
]


def normalize_text(x):
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_report_text(row):
    parts = []
    for c in REPORT_COLS:
        if c in row:
            v = normalize_text(row[c])
            if v:
                parts.append(v)
    return " | ".join(parts)


def find_matches(text):
    t = text.lower()
    labels = {k: 0 for k in TARGET_CLASSES}
    evidence = {k: [] for k in TARGET_CLASSES}

    for label, patterns in RULES.items():
        for pat in patterns:
            for m in re.finditer(pat, t, flags=re.IGNORECASE):
                span_start = max(0, m.start() - 50)
                span_end = min(len(text), m.end() + 80)
                snippet = text[span_start:span_end].strip()

                # Do not discard automatically; just mark if nearby negation/uncertainty appears.
                near_left = t[max(0, m.start() - 40):m.start()]
                negated_or_uncertain = any(re.search(npat, near_left, flags=re.IGNORECASE) for npat in NEGATION_PATTERNS)

                evidence[label].append({
                    "pattern": pat,
                    "match": m.group(0),
                    "snippet": snippet,
                    "negated_or_uncertain_context": bool(negated_or_uncertain),
                })
                labels[label] = 1

    return labels, evidence


def truncate(s, n=300):
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine-csv", default="data/raw/mimic_iv_ecg/machine_measurements.csv")
    ap.add_argument("--record-list", default="data/raw/mimic_iv_ecg/record_list.csv")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = full file")
    ap.add_argument("--sample-per-class", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    machine_csv = Path(args.machine_csv)
    record_list = Path(args.record_list)

    if not machine_csv.exists():
        raise FileNotFoundError(machine_csv)

    created = datetime.now(timezone.utc).isoformat()

    total_rows = 0
    rows_with_any_report = 0
    label_counts = Counter()
    neg_or_uncertain_counts = Counter()
    cooccur = defaultdict(Counter)
    keyword_support_rows = []
    sample_rows = []

    report_col_seen = set()

    usecols = None
    preview = pd.read_csv(machine_csv, nrows=1)
    available_cols = list(preview.columns)

    id_cols = [c for c in ["subject_id", "study_id", "cart_id", "ecg_time"] if c in available_cols]
    report_cols = [c for c in REPORT_COLS if c in available_cols]
    report_col_seen.update(report_cols)

    usecols = id_cols + report_cols

    for chunk_idx, chunk in enumerate(pd.read_csv(machine_csv, usecols=usecols, chunksize=args.chunksize, low_memory=False)):
        for _, row in chunk.iterrows():
            if args.max_rows and total_rows >= args.max_rows:
                break

            total_rows += 1
            report_text = build_report_text(row)

            if report_text:
                rows_with_any_report += 1

            labels, evidence = find_matches(report_text)
            positive = [k for k, v in labels.items() if v == 1]

            for label in positive:
                label_counts[label] += 1

                if any(ev.get("negated_or_uncertain_context") for ev in evidence[label]):
                    neg_or_uncertain_counts[label] += 1

            for a in positive:
                for b in positive:
                    cooccur[a][b] += 1

            if positive:
                for label in positive:
                    if sum(1 for r in sample_rows if r["candidate_label"] == label) < args.sample_per_class:
                        evs = evidence[label]
                        first_ev = evs[0] if evs else {}
                        sample_rows.append({
                            "candidate_label": label,
                            "subject_id": row.get("subject_id", ""),
                            "study_id": row.get("study_id", ""),
                            "cart_id": row.get("cart_id", ""),
                            "ecg_time": row.get("ecg_time", ""),
                            "matched_pattern": first_ev.get("pattern", ""),
                            "matched_text": first_ev.get("match", ""),
                            "negated_or_uncertain_context": first_ev.get("negated_or_uncertain_context", False),
                            "evidence_snippet": truncate(first_ev.get("snippet", ""), 500),
                            "report_text_preview": truncate(report_text, 500),
                        })

        if args.max_rows and total_rows >= args.max_rows:
            break

    # Build keyword support table
    for label in TARGET_CLASSES:
        keyword_support_rows.append({
            "target_superclass": label,
            "candidate_positive_count": int(label_counts[label]),
            "candidate_positive_pct": float(label_counts[label] / total_rows) if total_rows else 0.0,
            "negated_or_uncertain_context_count": int(neg_or_uncertain_counts[label]),
            "rule_count": len(RULES[label]),
            "rules": " | ".join(RULES[label]),
        })

    keyword_support_csv = out_dir / "full_mimic_report_keyword_support_v323.csv"
    with keyword_support_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "target_superclass",
            "candidate_positive_count",
            "candidate_positive_pct",
            "negated_or_uncertain_context_count",
            "rule_count",
            "rules",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(keyword_support_rows)

    candidates_csv = out_dir / "full_mimic_report_label_candidates_v323.csv"
    with candidates_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "candidate_label",
            "subject_id",
            "study_id",
            "cart_id",
            "ecg_time",
            "matched_pattern",
            "matched_text",
            "negated_or_uncertain_context",
            "evidence_snippet",
            "report_text_preview",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sample_rows)

    cooccur_matrix = {
        a: {b: int(cooccur[a][b]) for b in TARGET_CLASSES}
        for a in TARGET_CLASSES
    }

    record_list_info = {}
    if record_list.exists():
        try:
            rl = pd.read_csv(record_list, nrows=5)
            record_list_info = {
                "path": str(record_list),
                "columns": list(rl.columns),
                "preview_rows": int(len(rl)),
            }
        except Exception as e:
            record_list_info = {
                "path": str(record_list),
                "error": repr(e),
            }

    audit = {
        "project": "CardioTwin-AI",
        "version": "v3.2.3 Full MIMIC-IV-ECG label/report field audit",
        "created_at_utc": created,
        "machine_measurements_csv": str(machine_csv),
        "record_list_csv": str(record_list),
        "total_rows_scanned": int(total_rows),
        "rows_with_any_report_text": int(rows_with_any_report),
        "report_columns_detected": sorted(report_col_seen),
        "target_classes": TARGET_CLASSES,
        "candidate_label_support": {
            row["target_superclass"]: {
                "candidate_positive_count": row["candidate_positive_count"],
                "candidate_positive_pct": row["candidate_positive_pct"],
                "negated_or_uncertain_context_count": row["negated_or_uncertain_context_count"],
            }
            for row in keyword_support_rows
        },
        "cooccurrence_matrix": cooccur_matrix,
        "record_list_info": record_list_info,
        "outputs": {
            "keyword_support_csv": str(keyword_support_csv),
            "candidate_examples_csv": str(candidates_csv),
        },
        "interpretation": (
            "This is an automated report-field audit only. Candidate labels are weak labels and must be "
            "reviewed before freezing v3.2.4 mapping. Do not report external AUROC/AUPRC/F1 until mapping and waveform subset are frozen."
        ),
        "claim_boundary": "Metadata audit only. No model validation metrics computed.",
    }

    audit_json = out_dir / "full_mimic_report_field_audit_v323.json"
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    md = out_dir / "FULL_MIMIC_REPORT_FIELD_AUDIT_v323.md"
    md.write_text(f"""# CardioTwin-AI v3.2.3 Full MIMIC-IV-ECG Report Field Audit

Created: {created}

## Purpose

Audit Full MIMIC-IV-ECG machine report fields before freezing label mapping.

## Input

- machine_measurements.csv
- record_list.csv

## Rows scanned

- Total rows scanned: {total_rows}
- Rows with report text: {rows_with_any_report}
- Report columns detected: {", ".join(sorted(report_col_seen))}

## Candidate label support

{json.dumps(audit["candidate_label_support"], indent=2, ensure_ascii=False)}

## Co-occurrence matrix

{json.dumps(cooccur_matrix, indent=2, ensure_ascii=False)}

## Outputs

- `{keyword_support_csv}`
- `{candidates_csv}`
- `{audit_json}`

## Interpretation

This is an automated metadata audit only.

Candidate labels are weak labels. They must be reviewed and frozen before any external performance metric is reported.

## Next Step

Proceed to v3.2.4:

1. Review candidate examples.
2. Remove unsafe/ambiguous patterns.
3. Freeze report-to-superclass mapping.
4. Select waveform subset.
5. Run label-supported pilot validation.

## Claim Boundary

Metadata audit only. No AUROC/AUPRC/F1 computed.
""", encoding="utf-8")

    html = out_dir / "full_mimic_report_field_audit_v323.html"
    html.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CardioTwin-AI v3.2.3 Full MIMIC Report Field Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>CardioTwin-AI v3.2.3 Full MIMIC-IV-ECG Report Field Audit</h1>
  <div class="warning">
    Metadata audit only. Candidate labels are weak labels. No AUROC/AUPRC/F1 computed.
  </div>
  <h2>Audit Summary</h2>
  <pre>{json.dumps(audit, indent=2, ensure_ascii=False)}</pre>
</body>
</html>
""", encoding="utf-8")

    # Update v3.2 dataset plan
    plan_path = out_dir / "dataset_selection_plan_v32.json"
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            plan = {}
        plan["v3_2_3_report_field_audit"] = {
            "updated_at_utc": created,
            "status": "completed",
            "audit_json": str(audit_json),
            "keyword_support_csv": str(keyword_support_csv),
            "candidate_examples_csv": str(candidates_csv),
            "total_rows_scanned": int(total_rows),
            "rows_with_any_report_text": int(rows_with_any_report),
            "candidate_label_support": audit["candidate_label_support"],
        }
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DONE: v3.2.3 Full MIMIC-IV-ECG report field audit")
    print("AUDIT_JSON:", audit_json)
    print("KEYWORD_SUPPORT:", keyword_support_csv)
    print("CANDIDATE_EXAMPLES:", candidates_csv)
    print("MD:", md)
    print("HTML:", html)
    print(json.dumps({
        "total_rows_scanned": total_rows,
        "rows_with_any_report_text": rows_with_any_report,
        "candidate_label_support": audit["candidate_label_support"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
