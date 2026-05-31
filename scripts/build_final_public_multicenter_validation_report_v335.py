from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(".")
OUT = Path("artifacts/public_multicenter_validation_v33")
RELEASE = Path("artifacts/release_rc1")
OUT.mkdir(parents=True, exist_ok=True)
RELEASE.mkdir(parents=True, exist_ok=True)

FILES = {
    "cohort_summary": OUT / "public_locked_validation_cohort_summary_v330.json",
    "inference_summary": OUT / "public_locked_inference_summary_v331_full.json",
    "metrics_summary": OUT / "public_per_source_metrics_v332.json",
    "threshold_summary": OUT / "public_calibration_threshold_summary_v333.json",
    "failure_summary": OUT / "public_failure_case_review_summary_v334.json",
    "cohort_csv": OUT / "public_locked_validation_cohort_v330.csv",
    "predictions_csv": OUT / "public_locked_inference_predictions_v331_full.csv",
    "metrics_csv": OUT / "public_per_source_metrics_v332.csv",
    "threshold_curve_csv": OUT / "public_threshold_stress_curve_v333.csv",
    "failure_cases_csv": OUT / "public_failure_cases_v334.csv",
    "doctor_review_template": OUT / "doctor_in_the_loop_review_template_v334.csv",
}

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def fmt(x, digits=4):
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def build_results_table(metrics_json):
    group_summary = metrics_json.get("group_summary", {})
    rows = []

    preferred_order = [
        "cpsc_2018",
        "cpsc_2018_extra",
        "georgia",
        "ptb",
        "ALL_SOURCES_STACKED_REFERENCE_ONLY",
    ]

    for group in preferred_order:
        if group not in group_summary:
            continue
        g = group_summary[group]
        rows.append({
            "source_or_group": group,
            "macro_auroc": g.get("macro_auroc"),
            "macro_auprc": g.get("macro_auprc"),
            "macro_f1": g.get("macro_f1"),
            "macro_sensitivity": g.get("macro_sensitivity"),
            "macro_specificity": g.get("macro_specificity"),
            "macro_precision": g.get("macro_precision"),
            "labels_with_valid_auroc": g.get("labels_with_valid_auroc"),
            "claim_use": "descriptive_only_not_random_split" if group == "ALL_SOURCES_STACKED_REFERENCE_ONLY" else "source_separated_result",
        })

    return pd.DataFrame(rows)


def main():
    created = datetime.now(timezone.utc).isoformat()

    cohort = load_json(FILES["cohort_summary"])
    inference = load_json(FILES["inference_summary"])
    metrics = load_json(FILES["metrics_summary"])
    threshold = load_json(FILES["threshold_summary"])
    failure = load_json(FILES["failure_summary"])

    results_df = build_results_table(metrics)

    table_csv = OUT / "PUBLIC_PAPER_READY_RESULTS_TABLE_v335.csv"
    final_json = OUT / "public_multicenter_validation_final_summary_v335.json"
    final_md = OUT / "PUBLIC_MULTICENTER_VALIDATION_FINAL_REPORT_v335.md"
    final_html = OUT / "public_multicenter_validation_final_report_v335.html"
    claim_md = OUT / "PUBLIC_CLAIM_BOUNDARY_AND_NEXT_STEPS_v335.md"

    results_df.to_csv(table_csv, index=False, encoding="utf-8")

    source_table = cohort.get("source_table", {})
    label_totals = cohort.get("label_totals", {})
    group_summary = metrics.get("group_summary", {})
    failure_summary = failure.get("failure_summary", {})

    recommended_safe_claim = (
        "CardioTwin-AI was evaluated with a frozen runtime on a locked, source-separated "
        "public multi-center ECG cohort. The system showed promising discrimination across "
        "multiple public sources, with screening-oriented sensitivity and a need for "
        "source-aware threshold calibration and doctor-in-the-loop review before clinical claims."
    )

    disallowed_claims = [
        "clinically validated",
        "doctor-level diagnosis",
        "ready for autonomous clinical diagnosis",
        "generalizes to every hospital",
        "MIMIC-IV-ECG validated",
        "prospectively validated",
    ]

    final_summary = {
        "project": "CardioTwin-AI",
        "version": "v3.3.5 final public multicenter validation report pack",
        "created_at_utc": created,
        "scope": "Final report pack for v3.3.0-v3.3.4 public multi-center validation sequence.",
        "cohort": {
            "total_records_selected": cohort.get("total_records_selected"),
            "source_table": source_table,
            "label_totals": label_totals,
        },
        "inference": {
            "records_requested": inference.get("records_requested"),
            "ok_count": inference.get("ok_count"),
            "error_count": inference.get("error_count"),
            "runtime_seconds_total": inference.get("runtime_seconds_total"),
            "profile": inference.get("profile"),
            "device": inference.get("device"),
            "model_path": inference.get("model_path"),
            "threshold_path": inference.get("threshold_path"),
            "source_runtime": inference.get("source_runtime"),
        },
        "metrics": {
            "n_ok_rows": metrics.get("n_ok_rows"),
            "n_error_rows": metrics.get("n_error_rows"),
            "group_summary": group_summary,
            "interpretation": metrics.get("important_interpretation", []),
        },
        "calibration_threshold_stress": {
            "n_ok_rows": threshold.get("n_ok_rows"),
            "claim_boundary": threshold.get("claim_boundary"),
            "outputs": threshold.get("outputs", {}),
        },
        "failure_review": {
            "n_ok_rows": failure.get("n_ok_rows"),
            "failure_summary": failure_summary,
            "recommended_actions": failure.get("recommended_actions", []),
        },
        "safe_claim": recommended_safe_claim,
        "disallowed_claims": disallowed_claims,
        "claim_boundary": (
            "Research-use public multi-center validation evidence only. "
            "Not prospective clinical validation, not final diagnosis, and not clinical deployment."
        ),
        "outputs": {
            "paper_ready_results_table_csv": str(table_csv),
            "final_summary_json": str(final_json),
            "final_report_md": str(final_md),
            "final_report_html": str(final_html),
            "claim_boundary_md": str(claim_md),
        },
    }

    final_json.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = []
    md_lines.append("# CardioTwin-AI v3.3.5 Final Public Multicenter Validation Report")
    md_lines.append("")
    md_lines.append(f"Created: {created}")
    md_lines.append("")
    md_lines.append("## Executive Summary")
    md_lines.append("")
    md_lines.append(recommended_safe_claim)
    md_lines.append("")
    md_lines.append("## Locked Cohort")
    md_lines.append("")
    md_lines.append(f"- Total records selected: {cohort.get('total_records_selected')}")
    md_lines.append("- Sources:")
    for source, row in source_table.items():
        md_lines.append(f"  - {source}: {row.get('records')} records")
    md_lines.append("")
    md_lines.append("## Label Totals")
    md_lines.append("")
    for lab in TARGET_CLASSES:
        md_lines.append(f"- {lab}: {label_totals.get(lab)}")
    md_lines.append("")
    md_lines.append("## Frozen Inference")
    md_lines.append("")
    md_lines.append(f"- Records requested: {inference.get('records_requested')}")
    md_lines.append(f"- OK count: {inference.get('ok_count')}")
    md_lines.append(f"- Error count: {inference.get('error_count')}")
    md_lines.append(f"- Runtime seconds total: {fmt(inference.get('runtime_seconds_total'), 2)}")
    md_lines.append(f"- Device: {inference.get('device')}")
    md_lines.append(f"- Profile: {inference.get('profile')}")
    md_lines.append("")
    md_lines.append("## Source-separated Metrics")
    md_lines.append("")
    for _, r in results_df.iterrows():
        md_lines.append(f"### {r['source_or_group']}")
        md_lines.append("")
        md_lines.append(f"- Macro AUROC: {fmt(r['macro_auroc'])}")
        md_lines.append(f"- Macro AUPRC: {fmt(r['macro_auprc'])}")
        md_lines.append(f"- Macro F1: {fmt(r['macro_f1'])}")
        md_lines.append(f"- Macro sensitivity: {fmt(r['macro_sensitivity'])}")
        md_lines.append(f"- Macro specificity: {fmt(r['macro_specificity'])}")
        md_lines.append(f"- Macro precision: {fmt(r['macro_precision'])}")
        md_lines.append(f"- Claim use: {r['claim_use']}")
        md_lines.append("")
    md_lines.append("## Calibration and Threshold Stress")
    md_lines.append("")
    md_lines.append("v3.3.3 shows that screening thresholds favor sensitivity but increase false positives. Threshold recommendations are analytical only and do not modify the frozen runtime.")
    md_lines.append("")
    md_lines.append("## Failure-case Review")
    md_lines.append("")
    md_lines.append(f"- Total failure events: {failure_summary.get('total_failure_events')}")
    md_lines.append(f"- False positives: {failure_summary.get('failure_events_by_type', {}).get('false_positive')}")
    md_lines.append(f"- False negatives: {failure_summary.get('failure_events_by_type', {}).get('false_negative')}")
    md_lines.append(f"- High-confidence false positives: {failure_summary.get('high_confidence_false_positive_events')}")
    md_lines.append(f"- Low-score false negatives: {failure_summary.get('low_model_score_false_negative_events')}")
    md_lines.append(f"- Low-SQI failure events: {failure_summary.get('low_sqi_failure_events')}")
    md_lines.append("")
    md_lines.append("## Recommended Next Actions")
    md_lines.append("")
    for action in failure.get("recommended_actions", []):
        md_lines.append(f"- {action}")
    md_lines.append("- Build v3.4 source-aware calibration pack.")
    md_lines.append("- Run doctor-in-the-loop adjudication using the v3.3.4 template.")
    md_lines.append("- Keep MIMIC-IV-ECG as an access-gated future validation path.")
    md_lines.append("")
    md_lines.append("## Safe Claim")
    md_lines.append("")
    md_lines.append(recommended_safe_claim)
    md_lines.append("")
    md_lines.append("## Disallowed Claims")
    md_lines.append("")
    for c in disallowed_claims:
        md_lines.append(f"- {c}")
    md_lines.append("")
    md_lines.append("## Claim Boundary")
    md_lines.append("")
    md_lines.append(final_summary["claim_boundary"])

    final_md.write_text("\n".join(md_lines), encoding="utf-8")

    claim_md.write_text(
        "# CardioTwin-AI v3.3.5 Claim Boundary and Next Steps\n\n"
        f"Created: {created}\n\n"
        "## Safe Claim\n\n"
        + recommended_safe_claim
        + "\n\n## Do Not Claim Yet\n\n"
        + "\n".join("- " + c for c in disallowed_claims)
        + "\n\n## Required Before Clinical Claims\n\n"
        "- Prospective ECG collection.\n"
        "- Doctor-in-the-loop review.\n"
        "- Source-aware threshold calibration.\n"
        "- Safety/risk management documentation.\n"
        "- IRB/ethics workflow where applicable.\n"
        "- Independent external validation or access-gated MIMIC/KURIAS validation when available.\n",
        encoding="utf-8",
    )

    html_table = results_df.to_html(index=False)
    final_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CardioTwin-AI v3.3.5 Final Public Multicenter Validation Report</title>"
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45}"
        ".card{padding:14px;margin:12px 0;border:1px solid #ddd;border-radius:12px}"
        ".warning{padding:12px;background:#fff7ed;border-left:4px solid #f97316}"
        "table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border:1px solid #ddd;padding:6px}th{background:#f8fafc}"
        "pre{background:#f8fafc;padding:12px;overflow-x:auto}</style></head><body>"
        "<h1>CardioTwin-AI v3.3.5 Final Public Multicenter Validation Report</h1>"
        "<div class='warning'>Research-use public multicenter validation evidence only. Not clinical deployment.</div>"
        "<div class='card'><h2>Executive Summary</h2><p>" + recommended_safe_claim + "</p></div>"
        "<div class='card'><h2>Locked Inference</h2>"
        f"<p>Records: {inference.get('records_requested')} | OK: {inference.get('ok_count')} | Errors: {inference.get('error_count')} | Runtime: {fmt(inference.get('runtime_seconds_total'), 2)} sec</p>"
        "</div>"
        "<div class='card'><h2>Source-separated Metrics</h2>"
        + html_table +
        "</div>"
        "<div class='card'><h2>Failure Review</h2><pre>"
        + json.dumps(failure_summary, indent=2, ensure_ascii=False)
        + "</pre></div>"
        "<div class='card'><h2>Claim Boundary</h2><p>"
        + final_summary["claim_boundary"]
        + "</p></div>"
        "</body></html>",
        encoding="utf-8",
    )

    zip_path = RELEASE / "cardiotwin_v3_3_5_final_public_multicenter_validation_pack.zip"
    manifest_path = RELEASE / "cardiotwin_v3_3_5_final_public_multicenter_validation_manifest.json"

    include_files = [
        table_csv,
        final_json,
        final_md,
        final_html,
        claim_md,
        FILES["cohort_summary"],
        FILES["inference_summary"],
        FILES["metrics_summary"],
        FILES["threshold_summary"],
        FILES["failure_summary"],
        OUT / "PUBLIC_PER_SOURCE_METRICS_SUMMARY_v332.md",
        OUT / "PUBLIC_CALIBRATION_THRESHOLD_STRESS_v333.md",
        OUT / "PUBLIC_FAILURE_CASE_REVIEW_v334.md",
        OUT / "doctor_in_the_loop_review_template_v334.csv",
    ]

    include_files = [p for p in include_files if Path(p).exists()]

    manifest = {
        "project": "CardioTwin-AI",
        "version": "v3.3.5 Final Public Multicenter Validation Pack",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": final_summary,
        "files_indexed": len(include_files),
        "files": [
            {
                "path": Path(p).as_posix(),
                "size_bytes": int(Path(p).stat().st_size),
                "sha256": sha256_file(Path(p)),
            }
            for p in include_files
        ],
        "claim_boundary": final_summary["claim_boundary"],
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in include_files:
            z.write(p, Path(p).as_posix())
        z.write(manifest_path, manifest_path.as_posix())

    print("DONE: v3.3.5 final public multicenter validation report pack")
    print("RESULTS_TABLE:", table_csv)
    print("SUMMARY_JSON:", final_json)
    print("REPORT_MD:", final_md)
    print("REPORT_HTML:", final_html)
    print("CLAIM_BOUNDARY:", claim_md)
    print("ZIP:", zip_path)
    print("MANIFEST:", manifest_path)
    print(json.dumps({
        "total_records_selected": cohort.get("total_records_selected"),
        "ok_count": inference.get("ok_count"),
        "error_count": inference.get("error_count"),
        "failure_events": failure_summary.get("total_failure_events"),
        "files_indexed": manifest["files_indexed"],
        "claim_boundary": final_summary["claim_boundary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
