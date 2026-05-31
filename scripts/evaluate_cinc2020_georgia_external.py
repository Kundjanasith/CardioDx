from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from cardiotwin.constants import LEADS_12, PTBXL_SUPERCLASSES
from cardiotwin.signal.preprocessing import preprocess_ecg, pad_or_crop
from cardiotwin.signal.features import extract_features
from cardiotwin.models.baseline_ml import load_model, predict_proba
from cardiotwin.explain.lead_occlusion import align_feature_vector


# Direct high-confidence SNOMED mappings commonly appearing in CinC 2020.
DIRECT_CODE_TO_CLASS = {
    # Normal rhythm / normal ECG
    "426783006": "NORM",
    "426177001": "NORM",

    # MI / infarction families
    "164865005": "MI",
    "164866006": "MI",
    "164867002": "MI",
    "164868007": "MI",
    "57054005": "MI",

    # ST-T / repolarization
    "164930006": "STTC",
    "164931005": "STTC",
    "164934002": "STTC",
    "59931005": "STTC",
    "428750005": "STTC",

    # Conduction disturbance
    "270492004": "CD",
    "164909002": "CD",
    "59118001": "CD",
    "713426002": "CD",
    "713427006": "CD",
    "733534002": "CD",
    "6374002": "CD",
    "698252002": "CD",
    "445118002": "CD",
    "445211001": "CD",
    "251173003": "CD",
    "251170000": "CD",
    "111975006": "CD",

    # Hypertrophy
    "164873001": "HYP",
    "39732003": "HYP",
    "55930002": "HYP",
}


PATTERNS = [
    ("NORM", re.compile(r"\b(normal|sinus rhythm|normal sinus|nsr)\b", re.I)),
    ("MI", re.compile(r"\b(myocardial infarction|infarct|infarction|(^|[^a-z])mi([^a-z]|$)|ami|imi|asmi|almi|lmi|pmi)\b", re.I)),
    ("STTC", re.compile(r"\b(st depression|st elevation|st segment|t wave|t-wave|t abnormal|repolarization|sttc|std|ste|tab|tinv|nonspecific t)\b", re.I)),
    ("CD", re.compile(r"\b(block|bbb|rbbb|lbbb|av block|avb|ivcd|fascicular|lafb|lpfb|conduction|incomplete right bundle)\b", re.I)),
    ("HYP", re.compile(r"\b(hypertrophy|lvh|rvh|ventricular hypertrophy)\b", re.I)),
]


def build_code_to_class(cinc_root: Path) -> dict[str, str]:
    """
    Build SNOMED -> PTB-XL superclass mapping.

    Priority:
    1. configs/cinc2020_to_ptbxl_superclass_map_v21.csv
    2. configs/cinc2020_to_ptbxl_superclass_map.csv
    3. fallback direct mapping inside this script
    """
    config_candidates = [
        Path("configs/cinc2020_to_ptbxl_superclass_map_v21.csv"),
        Path("configs/cinc2020_to_ptbxl_superclass_map.csv"),
    ]

    for cfg in config_candidates:
        if cfg.exists():
            df = pd.read_csv(cfg)
            df["code"] = df["code"].astype(str)

            mapping = {}
            for _, row in df.iterrows():
                decision = str(row.get("decision", "")).strip()
                cls = str(row.get("ptbxl_superclass", "")).strip()
                code = str(row.get("code", "")).strip()

                if decision == "include" and cls in PTBXL_SUPERCLASSES and code:
                    mapping[code] = cls

            print(f"[INFO] Loaded harmonization map: {cfg} | mapped_codes={len(mapping)}")
            return mapping

    # Fallback only if no harmonization CSV is available.
    mapping = dict(DIRECT_CODE_TO_CLASS)

    for fname in ["dx_mapping_scored.csv", "dx_mapping_unscored.csv"]:
        path = cinc_root / fname
        if not path.exists():
            continue

        df = pd.read_csv(path)
        lower = {c.lower().strip(): c for c in df.columns}

        code_col = None
        for key in ["snomed ct code", "snomed_ct_code", "snomedctcode", "code"]:
            if key in lower:
                code_col = lower[key]
                break

        if code_col is None:
            continue

        text_cols = []
        for key in ["diagnosis", "abbreviation", "description", "dx"]:
            if key in lower:
                text_cols.append(lower[key])

        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            text = " ".join(str(row[c]) for c in text_cols if c in row.index)

            for cls, pat in PATTERNS:
                if pat.search(text):
                    mapping.setdefault(code, cls)
                    break

    print(f"[INFO] Loaded fallback direct mapping | mapped_codes={len(mapping)}")
    return mapping


def parse_dx(header_path: Path) -> list[str]:
    text = header_path.read_text(errors="ignore", encoding="utf-8-sig")

    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue

        # Accept all common variants:
        # #Dx: 426783006
        # # Dx: 426783006
        # #Dx : 426783006
        # # Diagnosis: 426783006
        body = s[1:].strip()
        key, sep, value = body.partition(":")
        if not sep:
            continue

        key = key.strip().lower()
        if key not in {"dx", "diagnosis"}:
            continue

        import re
        codes = re.findall(r"\d{6,}", value)
        if codes:
            return [c.strip() for c in codes if c.strip()]

        return [x.strip() for x in value.split(",") if x.strip()]

    return []


def parse_fs(header_path: Path) -> float:
    first = header_path.read_text(errors="ignore", encoding="utf-8").splitlines()[0]
    parts = first.split()
    if len(parts) >= 3:
        return float(parts[2])
    return 500.0


def parse_leads(header_path: Path) -> list[str]:
    lines = header_path.read_text(errors="ignore", encoding="utf-8").splitlines()
    leads = []
    for line in lines[1:]:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            lead = parts[-1].strip()
            if lead.upper() == "AVR":
                lead = "aVR"
            elif lead.upper() == "AVL":
                lead = "aVL"
            elif lead.upper() == "AVF":
                lead = "aVF"
            leads.append(lead)
    return leads if len(leads) == 12 else LEADS_12


def map_codes(codes: list[str], code_to_class: dict[str, str]) -> np.ndarray:
    y = np.zeros(len(PTBXL_SUPERCLASSES), dtype=int)
    for code in codes:
        cls = code_to_class.get(code)
        if cls in PTBXL_SUPERCLASSES:
            y[PTBXL_SUPERCLASSES.index(cls)] = 1
    return y


def load_signal(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path)
    x = np.asarray(data["val"], dtype=np.float32)
    if x.shape[0] == 12:
        x = x.T
    return x


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)

    overall = {}
    try:
        overall["auroc_macro"] = float(roc_auc_score(y_true, y_prob, average="macro"))
    except Exception:
        overall["auroc_macro"] = None
    try:
        overall["auprc_macro"] = float(average_precision_score(y_true, y_prob, average="macro"))
    except Exception:
        overall["auprc_macro"] = None

    overall["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    overall["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    overall["macro_recall_sensitivity"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    rows = []
    for j, label in enumerate(PTBXL_SUPERCLASSES):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        pp = y_prob[:, j]

        try:
            auroc = float(roc_auc_score(yt, pp))
        except Exception:
            auroc = None
        try:
            auprc = float(average_precision_score(yt, pp))
        except Exception:
            auprc = None

        rows.append({
            "label": label,
            "support": int(yt.sum()),
            "auroc": auroc,
            "auprc": auprc,
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "sensitivity": float(recall_score(yt, yp, zero_division=0)),
        })

    return overall, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cinc-root", default="data/raw/cinc2020")
    ap.add_argument("--subset", default="training/georgia")
    ap.add_argument("--model-path", default="artifacts/models/baseline_model.joblib")
    ap.add_argument("--out-dir", default="artifacts/external_validation/georgia_true_eval")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--target-fs", type=float, default=100.0)
    ap.add_argument("--duration-sec", type=float, default=10.0)
    args = ap.parse_args()

    cinc_root = Path(args.cinc_root)
    subset_dir = cinc_root / args.subset
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = sorted(subset_dir.rglob("*.hea"))
    if args.max_records:
        headers = headers[:args.max_records]

    if not headers:
        raise RuntimeError(f"No .hea files found under {subset_dir}")

    code_to_class = build_code_to_class(cinc_root)

    bundle = load_model(args.model_path)
    model = bundle["model"]

    X_rows, Y_rows, meta_rows, skipped = [], [], [], []
    raw_code_counter = Counter()
    mapped_code_counter = Counter()
    unmapped_code_counter = Counter()

    for i, hea in enumerate(headers, start=1):
        mat = hea.with_suffix(".mat")
        if not mat.exists():
            skipped.append({"record": str(hea), "reason": "missing_mat"})
            continue

        try:
            codes = parse_dx(hea)
            for c in codes:
                raw_code_counter[c] += 1

            y = map_codes(codes, code_to_class)

            if y.sum() == 0:
                for c in codes:
                    unmapped_code_counter[c] += 1
                skipped.append({"record": str(hea), "reason": "no_mapped_label", "dx": ",".join(codes)})
                continue

            for c in codes:
                if c in code_to_class:
                    mapped_code_counter[c] += 1

            fs = parse_fs(hea)
            leads = parse_leads(hea)
            sig = load_signal(mat)

            x, fs2 = preprocess_ecg(sig, fs, target_fs=args.target_fs, normalize=True)
            x = pad_or_crop(x, int(args.target_fs * args.duration_sec))

            feat, names, glob = extract_features(x, fs2, leads)
            feat = align_feature_vector(feat, names, bundle)

            X_rows.append(feat)
            Y_rows.append(y)
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
            skipped.append({"record": str(hea), "reason": f"error:{type(e).__name__}:{e}"})

    pd.DataFrame([
        {"code": k, "count": v, "mapped_class": code_to_class.get(k, "")}
        for k, v in raw_code_counter.most_common()
    ]).to_csv(out_dir / "georgia_dx_code_coverage.csv", index=False)

    pd.DataFrame(skipped).to_csv(out_dir / "georgia_records_skipped.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "georgia_records_used.csv", index=False)

    if not X_rows:
        top_unmapped = [
            {"code": k, "count": v}
            for k, v in unmapped_code_counter.most_common(50)
        ]
        summary = {
            "error": "No usable records after label harmonization.",
            "n_headers_seen": len(headers),
            "top_unmapped_codes": top_unmapped,
            "hint": "Open georgia_dx_code_coverage.csv and add direct code mappings.",
        }
        (out_dir / "georgia_external_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        raise RuntimeError("No usable Georgia records after label harmonization. See georgia_dx_code_coverage.csv")

    X = np.vstack(X_rows)
    Y = np.vstack(Y_rows)
    P = predict_proba(model, X)

    overall, per_class = compute_metrics(Y, P, threshold=0.5)
    per_class.to_csv(out_dir / "georgia_metrics_per_class.csv", index=False)

    np.save(out_dir / "Y_georgia.npy", Y)
    np.save(out_dir / "P_georgia.npy", P)

    label_counts = {PTBXL_SUPERCLASSES[j]: int(Y[:, j].sum()) for j in range(Y.shape[1])}

    summary = {
        "dataset": "PhysioNet_CinC_2020_Georgia",
        "subset": args.subset,
        "model_path": args.model_path,
        "n_header_files_seen": len(headers),
        "n_usable_records": int(Y.shape[0]),
        "n_skipped_records": len(skipped),
        "label_counts": label_counts,
        "overall_metrics_threshold_0p5": overall,
        "boundary": "True external-data processing for Georgia records with labels mapped to PTB-XL superclasses. Not official CinC challenge scoring.",
    }

    (out_dir / "georgia_external_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    html = f"""
    <html><head><meta charset="utf-8"><title>Georgia External Validation</title></head>
    <body>
    <h1>CardioTwin-AI Georgia External Validation</h1>
    <p><b>Dataset:</b> PhysioNet/CinC 2020 Georgia subset</p>
    <p><b>Usable records:</b> {Y.shape[0]} / {len(headers)}</p>
    <p><b>Skipped records:</b> {len(skipped)}</p>
    <h2>Overall metrics @ threshold 0.5</h2>
    <pre>{json.dumps(overall, indent=2)}</pre>
    <h2>Label counts</h2>
    <pre>{json.dumps(label_counts, indent=2)}</pre>
    <p>{summary["boundary"]}</p>
    </body></html>
    """
    (out_dir / "georgia_external_validation_report.html").write_text(html, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved Georgia external validation under: {out_dir}")


if __name__ == "__main__":
    main()
