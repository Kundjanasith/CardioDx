from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.io import loadmat

from cardiotwin.constants import LEADS_12, PTBXL_SUPERCLASSES
from cardiotwin.signal.preprocessing import preprocess_ecg, pad_or_crop
from cardiotwin.models.deep_ecg import make_deep_model


def load_baseline_georgia_helpers():
    script_path = Path("scripts/evaluate_cinc2020_georgia_external.py")
    spec = importlib.util.spec_from_file_location("geo_eval", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_deep_model(model_path: Path, device: str):
    ckpt = torch.load(model_path, map_location=device)
    model_name = ckpt.get("model_name", "inceptiontime")
    labels = ckpt.get("labels", PTBXL_SUPERCLASSES)

    model = make_deep_model(
        model_name,
        in_leads=12,
        n_classes=len(labels),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, model_name, labels


def predict_batches(model, X: np.ndarray, batch_size: int, device: str):
    ps = []
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            ps.append(prob)

    elapsed = time.perf_counter() - t0
    return np.vstack(ps), elapsed


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)

    def safe_float(fn):
        try:
            v = fn()
            if isinstance(v, float) and np.isnan(v):
                return None
            return float(v)
        except Exception:
            return None

    overall = {
        "auroc_macro": safe_float(lambda: roc_auc_score(y_true, y_prob, average="macro")),
        "auprc_macro": safe_float(lambda: average_precision_score(y_true, y_prob, average="macro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    rows = []
    for j, label in enumerate(PTBXL_SUPERCLASSES):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        pp = y_prob[:, j]

        rows.append({
            "label": label,
            "support": int(yt.sum()),
            "auroc": safe_float(lambda yt=yt, pp=pp: roc_auc_score(yt, pp)),
            "auprc": safe_float(lambda yt=yt, pp=pp: average_precision_score(yt, pp)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "sensitivity": float(recall_score(yt, yp, zero_division=0)),
        })

    per_class = pd.DataFrame(rows)

    valid = per_class[per_class["support"] >= 20].copy()
    valid_summary = {
        "valid_support_threshold": 20,
        "valid_labels": valid["label"].tolist(),
        "excluded_labels": per_class[per_class["support"] < 20]["label"].tolist(),
        "macro_auroc_valid": float(valid["auroc"].dropna().mean()),
        "macro_auprc_valid": float(valid["auprc"].dropna().mean()),
        "macro_f1_valid": float(valid["f1"].mean()),
        "macro_precision_valid": float(valid["precision"].mean()),
        "macro_sensitivity_valid": float(valid["sensitivity"].mean()),
        "interpretation": "Valid-label macro excludes classes with insufficient positive support for stable external metrics.",
    }

    return overall, per_class, valid_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cinc-root", default="data/raw/cinc2020")
    ap.add_argument("--subset", default="training/georgia")
    ap.add_argument("--model-path", default="artifacts/deep_models/inceptiontime_model.pt")
    ap.add_argument("--out-dir", default="artifacts/external_validation/georgia_deep_inceptiontime_v21")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--target-fs", type=float, default=100.0)
    ap.add_argument("--duration-sec", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    geo = load_baseline_georgia_helpers()

    cinc_root = Path(args.cinc_root)
    subset_dir = cinc_root / args.subset
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = sorted(subset_dir.rglob("*.hea"))
    if args.max_records:
        headers = headers[:args.max_records]

    if not headers:
        raise RuntimeError(f"No .hea files found under {subset_dir}")

    code_to_class = geo.build_code_to_class(cinc_root)

    model, model_name, labels = load_deep_model(Path(args.model_path), args.device)

    X_rows = []
    Y_rows = []
    meta_rows = []
    skipped = []

    for i, hea in enumerate(headers, start=1):
        mat = hea.with_suffix(".mat")
        if not mat.exists():
            skipped.append({"record": str(hea), "reason": "missing_mat"})
            continue

        try:
            codes = geo.parse_dx(hea)
            y = geo.map_codes(codes, code_to_class)

            if y.sum() == 0:
                skipped.append({
                    "record": str(hea),
                    "reason": "no_mapped_label",
                    "dx": ",".join(codes),
                })
                continue

            fs = geo.parse_fs(hea)
            sig = geo.load_signal(mat)

            x, fs2 = preprocess_ecg(sig, fs, target_fs=args.target_fs, normalize=True)
            x = pad_or_crop(x, int(args.target_fs * args.duration_sec))

            # Deep model expects [leads, samples]
            x = x.astype(np.float32).T

            X_rows.append(x)
            Y_rows.append(y.astype(np.float32))
            meta_rows.append({
                "record_id": hea.stem,
                "hea_path": str(hea),
                "dx_codes": ",".join(codes),
                "mapped_labels": ",".join([PTBXL_SUPERCLASSES[k] for k, v in enumerate(y) if v == 1]),
                "fs": fs,
            })

            if i % 250 == 0:
                print(f"processed {i}/{len(headers)} | usable={len(Y_rows)} skipped={len(skipped)}")

        except Exception as e:
            skipped.append({
                "record": str(hea),
                "reason": f"error:{type(e).__name__}:{e}",
            })

    if not X_rows:
        raise RuntimeError("No usable Georgia records for deep external evaluation.")

    X = np.stack(X_rows, axis=0)
    Y = np.vstack(Y_rows).astype(np.float32)

    P, elapsed = predict_batches(model, X, args.batch_size, args.device)

    overall, per_class, valid_summary = compute_metrics(Y, P, threshold=0.5)

    per_class.to_csv(out_dir / "georgia_deep_metrics_per_class.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "georgia_deep_records_used.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "georgia_deep_records_skipped.csv", index=False)
    np.save(out_dir / "Y_georgia_deep.npy", Y)
    np.save(out_dir / "P_georgia_deep.npy", P)

    label_counts = {PTBXL_SUPERCLASSES[j]: int(Y[:, j].sum()) for j in range(Y.shape[1])}

    summary = {
        "dataset": "PhysioNet_CinC_2020_Georgia",
        "subset": args.subset,
        "model_type": "deep_ecg",
        "model_name": model_name,
        "model_path": args.model_path,
        "device": args.device,
        "n_header_files_seen": len(headers),
        "n_usable_records": int(Y.shape[0]),
        "n_skipped_records": len(skipped),
        "label_counts": label_counts,
        "overall_metrics_threshold_0p5": overall,
        "valid_label_metrics": valid_summary,
        "inference_eval_seconds_total": float(elapsed),
        "inference_latency_ms_per_record": float((elapsed / max(len(Y), 1)) * 1000.0),
        "boundary": "True external-data processing for Georgia records with v2.1 harmonization. Not official CinC challenge scoring.",
    }

    (out_dir / "georgia_deep_external_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out_dir / "georgia_deep_valid_label_metrics.json").write_text(
        json.dumps(valid_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    html = f"""
    <html><head><meta charset="utf-8"><title>Georgia Deep External Validation</title></head>
    <body>
    <h1>CardioTwin-AI Georgia Deep External Validation</h1>
    <p><b>Dataset:</b> PhysioNet/CinC 2020 Georgia subset</p>
    <p><b>Model:</b> {model_name}</p>
    <p><b>Usable records:</b> {Y.shape[0]} / {len(headers)}</p>
    <p><b>Skipped records:</b> {len(skipped)}</p>
    <p><b>Latency:</b> {summary["inference_latency_ms_per_record"]:.3f} ms/record on {args.device}</p>
    <h2>Overall metrics @ threshold 0.5</h2>
    <pre>{json.dumps(overall, indent=2)}</pre>
    <h2>Valid-label metrics</h2>
    <pre>{json.dumps(valid_summary, indent=2)}</pre>
    <h2>Label counts</h2>
    <pre>{json.dumps(label_counts, indent=2)}</pre>
    <p>{summary["boundary"]}</p>
    </body></html>
    """
    (out_dir / "georgia_deep_external_validation_report.html").write_text(html, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved Georgia deep external validation under: {out_dir}")


if __name__ == "__main__":
    main()
