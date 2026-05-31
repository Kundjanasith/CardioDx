from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from cardiotwin.constants import PTBXL_SUPERCLASSES
from cardiotwin.data.ptbxl_loader import iter_ptbxl_records, ptbxl_split_from_fold, write_manifest
from cardiotwin.signal.preprocessing import preprocess_ecg, pad_or_crop
from cardiotwin.signal.features import extract_features
from cardiotwin.signal.quality import compute_sqi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl-root", required=True)
    ap.add_argument("--out-dir", default="artifacts/processed")
    ap.add_argument("--sampling-rate", type=float, default=100.0)
    ap.add_argument("--duration-sec", type=float, default=10.0)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--use-hr", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_dir = out / "records_npz"
    records_dir.mkdir(exist_ok=True)

    X, Y, ids, splits, sqis = [], [], [], [], []
    feature_names = None
    index_rows = []
    meta_manifest = []
    for rec in iter_ptbxl_records(args.ptbxl_root, use_lr=not args.use_hr, max_records=args.max_records):
        x, fs = preprocess_ecg(rec.signal, rec.fs, target_fs=args.sampling_rate)
        x = pad_or_crop(x, int(args.sampling_rate * args.duration_sec))
        sqi = compute_sqi(x, fs, rec.leads)
        feat, names, glob = extract_features(x, fs, rec.leads)
        if feature_names is None:
            feature_names = names
        y = np.array([rec.labels.get(c, 0) for c in PTBXL_SUPERCLASSES], dtype=np.int64)
        fold = rec.metadata.get("strat_fold", -1)
        split = ptbxl_split_from_fold(fold)
        npz_path = records_dir / f"{rec.record_id}.npz"
        np.savez_compressed(npz_path, signal=x.astype(np.float32), fs=fs, leads=np.array(rec.leads),
                            labels=y, record_id=rec.record_id, split=split)
        X.append(feat); Y.append(y); ids.append(rec.record_id); splits.append(split); sqis.append(sqi["overall_sqi"])
        row = {"record_id": rec.record_id, "split": split, "fs": fs, "npz_path": str(npz_path), "sqi": sqi["overall_sqi"]}
        for c, val in zip(PTBXL_SUPERCLASSES, y): row[c] = int(val)
        index_rows.append(row)
        meta_manifest.append({"record_id": rec.record_id, "split": split, "labels": rec.labels, "metadata": rec.metadata})

    X = np.vstack(X).astype(np.float32)
    Y = np.vstack(Y).astype(np.int64)
    np.save(out / "X_features.npy", X)
    np.save(out / "Y_labels.npy", Y)
    pd.DataFrame(index_rows).to_csv(out / "records_index.csv", index=False)
    (out / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    write_manifest(meta_manifest, out / "manifest.json")
    print(f"Saved processed dataset to {out}")
    print(f"X: {X.shape}, Y: {Y.shape}")

if __name__ == "__main__":
    main()
