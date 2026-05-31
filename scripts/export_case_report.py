from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from cardiotwin.models.baseline_ml import load_model
from cardiotwin.pipeline.inference import run_inference
from cardiotwin.reports.report_generator import generate_html_report, save_json_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="artifacts/processed")
    ap.add_argument("--model-path", default="artifacts/models/baseline_model.joblib")
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--out-dir", default="artifacts/reports")
    args = ap.parse_args()
    p = Path(args.processed_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(p / "records_index.csv")
    row = idx.iloc[args.record_index]
    data = np.load(row["npz_path"], allow_pickle=True)
    signal = data["signal"]
    fs = float(data["fs"])
    leads = [str(x) for x in data["leads"]]
    bundle = load_model(args.model_path)
    state = run_inference(bundle, signal, fs, leads, record_id=str(row["record_id"]))
    save_json_report(state, out / f"case_{row['record_id']}.json")
    generate_html_report(state, out / f"case_{row['record_id']}.html")
    print(f"Saved report under {out}")

if __name__ == "__main__":
    main()
