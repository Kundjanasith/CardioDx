from __future__ import annotations
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from cardiotwin.constants import LEADS_12, PTBXL_SUPERCLASSES
from cardiotwin.data.schema import ECGRecord


def load_ptbxl_metadata(ptbxl_root: str | Path) -> pd.DataFrame:
    root = Path(ptbxl_root)
    csv_path = root / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Download PTB-XL and place it under data/raw/ptbxl.")
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def load_scp_statements(ptbxl_root: str | Path) -> pd.DataFrame:
    root = Path(ptbxl_root)
    path = root / "scp_statements.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    return pd.read_csv(path, index_col=0)


def scp_to_superclasses(scp_codes: Dict[str, float], scp_df: pd.DataFrame, min_confidence: float = 0.0) -> Dict[str, int]:
    labels = {c: 0 for c in PTBXL_SUPERCLASSES}
    for code, confidence in scp_codes.items():
        if confidence < min_confidence or code not in scp_df.index:
            continue
        row = scp_df.loc[code]
        if bool(row.get("diagnostic", 0)):
            superclass = row.get("diagnostic_class", None)
            if isinstance(superclass, str) and superclass in labels:
                labels[superclass] = 1
    return labels


def read_ptbxl_record(ptbxl_root: str | Path, row: pd.Series, use_lr: bool = True) -> Tuple[np.ndarray, float, List[str]]:
    root = Path(ptbxl_root)
    filename = row["filename_lr"] if use_lr else row["filename_hr"]
    record_path = root / filename
    signal, fields = wfdb.rdsamp(str(record_path))
    fs = float(fields.get("fs", 100 if use_lr else 500))
    leads = fields.get("sig_name", LEADS_12)
    # Reorder to standard if possible
    if set(LEADS_12).issubset(set(leads)):
        idx = [leads.index(l) for l in LEADS_12]
        signal = signal[:, idx]
        leads = LEADS_12
    return signal.astype(np.float32), fs, list(leads)


def iter_ptbxl_records(ptbxl_root: str | Path, use_lr: bool = True, max_records: Optional[int] = None,
                       min_confidence: float = 0.0):
    df = load_ptbxl_metadata(ptbxl_root)
    scp_df = load_scp_statements(ptbxl_root)
    if max_records:
        df = df.iloc[:max_records]
    for ecg_id, row in tqdm(df.iterrows(), total=len(df), desc="Reading PTB-XL"):
        try:
            signal, fs, leads = read_ptbxl_record(ptbxl_root, row, use_lr=use_lr)
            labels = scp_to_superclasses(row["scp_codes"], scp_df, min_confidence=min_confidence)
            metadata = {
                "ecg_id": int(ecg_id),
                "patient_id": row.get("patient_id", None),
                "age": row.get("age", None),
                "sex": row.get("sex", None),
                "strat_fold": int(row.get("strat_fold", -1)),
                "filename_lr": row.get("filename_lr", None),
                "filename_hr": row.get("filename_hr", None),
                "source": "PTB-XL",
            }
            yield ECGRecord(str(ecg_id), signal, fs, leads, labels, metadata)
        except Exception as e:
            print(f"[WARN] failed record {ecg_id}: {e}")


def write_manifest(records_meta: List[Dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = {c: 0 for c in PTBXL_SUPERCLASSES}
    split_counts = {"train": 0, "val": 0, "test": 0}
    for m in records_meta:
        for c, v in m.get("labels", {}).items():
            label_counts[c] = label_counts.get(c, 0) + int(v)
        split = m.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    manifest = {
        "project": "CardioTwin-AI 12L",
        "n_records": len(records_meta),
        "classes": PTBXL_SUPERCLASSES,
        "class_counts": label_counts,
        "split_counts": split_counts,
        "records": records_meta[:10],
        "note": "Only first 10 record metadata items are stored here for compactness. Full index is records_index.csv."
    }
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def ptbxl_split_from_fold(strat_fold: int) -> str:
    if strat_fold == 10:
        return "test"
    if strat_fold == 9:
        return "val"
    return "train"
