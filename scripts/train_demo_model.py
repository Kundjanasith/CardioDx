from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from cardiotwin.constants import LEADS_12, PTBXL_SUPERCLASSES
from cardiotwin.signal.features import extract_features
from cardiotwin.models.baseline_ml import train_baseline, save_model, predict_proba
from cardiotwin.models.metrics import classification_metrics, save_metrics
from cardiotwin.reports.model_card import write_model_card
from scripts.make_synthetic_demo import synthetic_ecg


def make_case(seed: int, cls: str):
    sig, _ = synthetic_ecg(seed=seed, noise=0.02)
    # inject simple class-specific signatures for demo only
    if cls == "MI":
        for lead in ["II", "III", "aVF"]:
            sig[:, LEADS_12.index(lead)] += 0.25
    elif cls == "STTC":
        for lead in ["V3", "V4", "V5"]:
            sig[:, LEADS_12.index(lead)] -= 0.18
    elif cls == "CD":
        sig = np.roll(sig, 4, axis=0) * 0.8 + sig * 0.2
    elif cls == "HYP":
        for lead in ["I", "aVL", "V5", "V6"]:
            sig[:, LEADS_12.index(lead)] *= 1.6
    y = np.zeros(len(PTBXL_SUPERCLASSES), dtype=int)
    y[PTBXL_SUPERCLASSES.index(cls)] = 1
    if cls != "NORM":
        y[PTBXL_SUPERCLASSES.index("NORM")] = 0
    return sig.astype('float32'), y


def main():
    out = Path("artifacts/models"); out.mkdir(parents=True, exist_ok=True)
    met = Path("artifacts/metrics/demo"); met.mkdir(parents=True, exist_ok=True)
    X, Y = [], []
    names = None
    classes = PTBXL_SUPERCLASSES
    for i in range(250):
        cls = classes[i % len(classes)]
        sig, y = make_case(1000+i, cls)
        feat, names, _ = extract_features(sig, 500, LEADS_12)
        X.append(feat); Y.append(y)
    X = np.vstack(X); Y = np.vstack(Y)
    model = train_baseline(X[:200], Y[:200], model_type="logreg")
    prob = predict_proba(model, X[200:])
    overall, per_class = classification_metrics(Y[200:], prob, label_names=PTBXL_SUPERCLASSES)
    overall["warning"] = "DEMO MODEL trained on synthetic data only. Replace by running PTB-XL pipeline before any research use."
    save_metrics(overall, per_class, met)
    save_model(model, out / "baseline_model.joblib", names, PTBXL_SUPERCLASSES)
    write_model_card(out / "model_card.json", overall, {"dataset": "synthetic demo only", "n_records": 250})
    (out / "DEMO_MODEL_NOTICE.txt").write_text("This baseline_model.joblib is a synthetic demo placeholder so the dashboard runs immediately. Train on PTB-XL with scripts/run_reproducible_baseline.py for research results.\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))
    print(f"Saved demo model to {out/'baseline_model.joblib'}")

if __name__ == "__main__":
    main()
