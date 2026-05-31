from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from cardiotwin.constants import PTBXL_SUPERCLASSES
from cardiotwin.models.baseline_ml import load_model, predict_proba
from cardiotwin.models.metrics import classification_metrics, save_metrics
from cardiotwin.eval.safety import safety_metrics
from cardiotwin.eval.efficiency import benchmark_inference, file_size_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="artifacts/processed")
    ap.add_argument("--model-path", default="artifacts/models/baseline_model.joblib")
    ap.add_argument("--out-dir", default="artifacts/metrics")
    args = ap.parse_args()
    p = Path(args.processed_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X = np.load(p / "X_features.npy"); Y = np.load(p / "Y_labels.npy")
    idx = pd.read_csv(p / "records_index.csv")
    test_mask = idx["split"].values == "test"
    if test_mask.sum() == 0:
        print("[WARN] no test split found; evaluating on all data")
        test_mask = np.ones(len(idx), dtype=bool)
    bundle = load_model(args.model_path); model = bundle["model"]
    prob = predict_proba(model, X[test_mask])
    overall, per_class = classification_metrics(Y[test_mask], prob, label_names=PTBXL_SUPERCLASSES)
    eff = benchmark_inference(lambda z: predict_proba(model, z), X[test_mask][:1], n_runs=30)
    eff["model_size_mb"] = file_size_mb(args.model_path)
    safety = safety_metrics(idx.loc[test_mask, "sqi"].tolist(), prob, Y[test_mask])
    all_metrics = {**overall, **{f"efficiency_{k}": v for k, v in eff.items()}, **safety}
    save_metrics(all_metrics, per_class, out)
    print(json.dumps(all_metrics, indent=2))

if __name__ == "__main__":
    main()
