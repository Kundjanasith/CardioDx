from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from cardiotwin.constants import PTBXL_SUPERCLASSES
from cardiotwin.models.baseline_ml import train_baseline, save_model, predict_proba
from cardiotwin.models.metrics import classification_metrics, save_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="artifacts/processed")
    ap.add_argument("--out-dir", default="artifacts/models")
    ap.add_argument("--model-type", default="logreg", choices=["logreg", "rf"])
    args = ap.parse_args()
    p = Path(args.processed_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X = np.load(p / "X_features.npy")
    Y = np.load(p / "Y_labels.npy")
    idx = pd.read_csv(p / "records_index.csv")
    train_mask = idx["split"].values == "train"
    val_mask = idx["split"].values == "val"
    feature_names = json.loads((p / "feature_names.json").read_text(encoding="utf-8"))
    model = train_baseline(X[train_mask], Y[train_mask], model_type=args.model_type)
    save_model(model, out / "baseline_model.joblib", feature_names, PTBXL_SUPERCLASSES)
    if val_mask.sum() > 0:
        prob = predict_proba(model, X[val_mask])
        overall, per_class = classification_metrics(Y[val_mask], prob, label_names=PTBXL_SUPERCLASSES)
        save_metrics(overall, per_class, out / "validation_metrics")
        print(json.dumps(overall, indent=2))
    print(f"Saved model to {out / 'baseline_model.joblib'}")

if __name__ == "__main__":
    main()
