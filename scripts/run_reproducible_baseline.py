from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from datetime import datetime
from cardiotwin.reports.model_card import write_model_card


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl-root", required=True)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--sampling-rate", type=float, default=100.0)
    args = ap.parse_args()
    py = sys.executable
    processed = Path("artifacts/processed")
    models = Path("artifacts/models")
    metrics = Path("artifacts/metrics")
    reports = Path("artifacts/reports")
    exp = Path("artifacts/experiments") / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    exp.mkdir(parents=True, exist_ok=True)
    cmd = [py, "scripts/prepare_ptbxl.py", "--ptbxl-root", args.ptbxl_root, "--out-dir", str(processed), "--sampling-rate", str(args.sampling_rate)]
    if args.max_records:
        cmd += ["--max-records", str(args.max_records)]
    run(cmd)
    run([py, "scripts/train_baseline.py", "--processed-dir", str(processed), "--out-dir", str(models)])
    run([py, "scripts/evaluate_model.py", "--processed-dir", str(processed), "--model-path", str(models / "baseline_model.joblib"), "--out-dir", str(metrics)])
    run([py, "scripts/export_case_report.py", "--processed-dir", str(processed), "--model-path", str(models / "baseline_model.joblib"), "--record-index", "0", "--out-dir", str(reports)])
    metrics_json = json.loads((metrics / "metrics_overall.json").read_text(encoding="utf-8"))
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    write_model_card(models / "model_card.json", metrics_json, manifest)
    exp_log = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "ptbxl_root": args.ptbxl_root,
        "max_records": args.max_records,
        "sampling_rate": args.sampling_rate,
        "processed_dir": str(processed),
        "model_path": str(models / "baseline_model.joblib"),
        "metrics_dir": str(metrics),
        "reports_dir": str(reports),
        "status": "completed",
    }
    (exp / "experiment_log.json").write_text(json.dumps(exp_log, indent=2), encoding="utf-8")
    print(f"\nDONE. Experiment log: {exp / 'experiment_log.json'}")

if __name__ == "__main__":
    main()
