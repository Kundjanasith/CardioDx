from pathlib import Path
import argparse
import csv
import json
import time
import traceback
import zipfile
import hashlib
from datetime import datetime, timezone
import importlib.util

import pandas as pd

ROOT = Path(".")
OUT = Path("artifacts/public_multicenter_validation_v33")
RELEASE = Path("artifacts/release_rc1")
OUT.mkdir(parents=True, exist_ok=True)
RELEASE.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

COHORT_CSV = OUT / "public_locked_validation_cohort_v330.csv"
BRIDGE_PATH = Path("src/cardiotwin/runtime/v304_real_inference_bridge.py")

MODEL_PATH = Path("artifacts/models/inceptiontime_v21_safety.pt")
THRESHOLD_PATH = Path("artifacts/deep_safety_v21/threshold_profiles_deep.json")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bridge():
    if not BRIDGE_PATH.exists():
        raise FileNotFoundError(BRIDGE_PATH)

    spec = importlib.util.spec_from_file_location("v304_real_inference_bridge", BRIDGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    required = ["load_wfdb_hea_mat", "run_v304_real_inference"]
    for name in required:
        if not hasattr(mod, name):
            raise RuntimeError(f"Bridge missing required function: {name}")

    return mod


def get_prob(result, label):
    probs = result.get("probabilities", {}) or {}
    try:
        return float(probs.get(label, "nan"))
    except Exception:
        return float("nan")


def get_threshold(result, label):
    th = result.get("thresholds", {}) or {}
    try:
        return float(th.get(label, "nan"))
    except Exception:
        return float("nan")


def boolish(x):
    if isinstance(x, bool):
        return x
    if str(x).strip().lower() in {"true", "1", "yes"}:
        return True
    return False


def safe_json_dumps(x):
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return json.dumps(str(x), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--limit", type=int, default=20, help="0 = full cohort")
    ap.add_argument("--profile", default="screening")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()

    created = datetime.now(timezone.utc).isoformat()

    cohort_path = Path(args.cohort)
    if not cohort_path.exists():
        raise FileNotFoundError(cohort_path)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(THRESHOLD_PATH)

    bridge = load_bridge()

    df = pd.read_csv(cohort_path)

    if args.start > 0:
        df = df.iloc[args.start:].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    mode = "smoke" if args.limit and args.limit > 0 else "full"
    out_csv = OUT / f"public_locked_inference_predictions_v331_{mode}.csv"
    out_json = OUT / f"public_locked_inference_summary_v331_{mode}.json"
    out_errors = OUT / f"public_locked_inference_errors_v331_{mode}.json"

    rows = []
    errors = []

    t0 = time.time()

    print("CardioTwin-AI v3.3.1 locked public inference")
    print("mode:", mode)
    print("records_to_run:", len(df))
    print("profile:", args.profile)
    print("device:", args.device)
    print("model:", MODEL_PATH)
    print("threshold:", THRESHOLD_PATH)
    print("output:", out_csv)

    for idx, row in df.iterrows():
        rec_start = time.time()

        source_id = str(row.get("source_id", ""))
        record_id = str(row.get("record_id", ""))
        hea_path = Path(str(row.get("hea_path", "")))

        base_out = {
            "row_index": int(idx),
            "source_id": source_id,
            "record_id": record_id,
            "hea_path": str(hea_path),
            "true_NORM": int(row.get("label_NORM", 0)),
            "true_MI": int(row.get("label_MI", 0)),
            "true_STTC": int(row.get("label_STTC", 0)),
            "true_CD": int(row.get("label_CD", 0)),
            "true_HYP": int(row.get("label_HYP", 0)),
            "metric_labels": str(row.get("metric_labels", "")),
            "status": "error",
            "error": "",
        }

        try:
            if not hea_path.exists():
                raise FileNotFoundError(hea_path)

            x_raw, fs, source_meta_loaded = bridge.load_wfdb_hea_mat(hea_path)

            source_meta = {
                "source_id": source_id,
                "record_id": record_id,
                "cohort_row_index": int(idx),
                "hea_path": str(hea_path),
                "mat_path": str(row.get("mat_path", "")),
                "dat_path": str(row.get("dat_path", "")),
                "metric_labels": str(row.get("metric_labels", "")),
            }

            if isinstance(source_meta_loaded, dict):
                source_meta.update(source_meta_loaded)

            result = bridge.run_v304_real_inference(
                x_raw=x_raw,
                fs=float(fs),
                model_path=MODEL_PATH,
                threshold_path=THRESHOLD_PATH,
                profile=args.profile,
                device=args.device,
                source_meta=source_meta,
            )

            positive_labels = result.get("positive_labels", []) or []
            abnormal_positive_labels = result.get("abnormal_positive_labels", []) or []

            out = dict(base_out)
            out.update({
                "status": "ok",
                "error": "",
                "raw_fs": result.get("raw_fs", fs),
                "ai_fs": result.get("ai_fs", ""),
                "sqi": result.get("sqi", ""),
                "low_sqi": result.get("low_sqi", ""),
                "uncertain": result.get("uncertain", ""),
                "recommendation": result.get("recommendation", ""),
                "threshold_profile": result.get("threshold_profile", args.profile),
                "threshold_source": result.get("threshold_source", ""),
                "inference_mode": result.get("inference_mode", ""),
                "inference_error": result.get("inference_error", ""),
                "positive_labels": "|".join(map(str, positive_labels)),
                "abnormal_positive_labels": "|".join(map(str, abnormal_positive_labels)),
                "pred_NORM": int("NORM" in positive_labels),
                "pred_MI": int("MI" in positive_labels),
                "pred_STTC": int("STTC" in positive_labels),
                "pred_CD": int("CD" in positive_labels),
                "pred_HYP": int("HYP" in positive_labels),
                "prob_NORM": get_prob(result, "NORM"),
                "prob_MI": get_prob(result, "MI"),
                "prob_STTC": get_prob(result, "STTC"),
                "prob_CD": get_prob(result, "CD"),
                "prob_HYP": get_prob(result, "HYP"),
                "threshold_NORM": get_threshold(result, "NORM"),
                "threshold_MI": get_threshold(result, "MI"),
                "threshold_STTC": get_threshold(result, "STTC"),
                "threshold_CD": get_threshold(result, "CD"),
                "threshold_HYP": get_threshold(result, "HYP"),
                "region_summary_json": safe_json_dumps(result.get("region_summary", {})),
                "region_mapper_meta_json": safe_json_dumps(result.get("region_mapper_meta", {})),
                "runtime_seconds": time.time() - rec_start,
            })
            rows.append(out)

        except Exception as e:
            err = dict(base_out)
            err.update({
                "status": "error",
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.time() - rec_start,
            })
            rows.append(err)
            errors.append(err)

        if len(rows) % args.save_every == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
            print(f"[SAVE] {len(rows)}/{len(df)} rows -> {out_csv}")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_csv, index=False, encoding="utf-8")

    ok_count = int((pred_df["status"] == "ok").sum()) if "status" in pred_df else 0
    error_count = int((pred_df["status"] == "error").sum()) if "status" in pred_df else len(errors)

    source_runtime = {}
    if len(pred_df) > 0 and "source_id" in pred_df:
        for source_id, g in pred_df.groupby("source_id"):
            source_runtime[source_id] = {
                "rows": int(len(g)),
                "ok": int((g["status"] == "ok").sum()),
                "error": int((g["status"] == "error").sum()),
                "mean_runtime_seconds": float(pd.to_numeric(g["runtime_seconds"], errors="coerce").mean()),
            }

    summary = {
        "project": "CardioTwin-AI",
        "version": f"v3.3.1 public locked inference {mode}",
        "created_at_utc": created,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "cohort_csv": str(cohort_path),
        "predictions_csv": str(out_csv),
        "records_requested": int(len(df)),
        "ok_count": ok_count,
        "error_count": error_count,
        "runtime_seconds_total": time.time() - t0,
        "profile": args.profile,
        "device": args.device,
        "model_path": str(MODEL_PATH),
        "threshold_path": str(THRESHOLD_PATH),
        "source_runtime": source_runtime,
        "claim_boundary": "Frozen runtime inference only. Metrics are computed in v3.3.2, not in this step.",
    }

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    out_errors.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DONE: v3.3.1 locked inference")
    print("PREDICTIONS:", out_csv)
    print("SUMMARY:", out_json)
    print("ERRORS:", out_errors)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if mode == "full":
        zip_path = RELEASE / "cardiotwin_v3_3_1_public_locked_inference_pack.zip"
        manifest_path = RELEASE / "cardiotwin_v3_3_1_public_locked_inference_manifest.json"

        files = [out_csv, out_json, out_errors, OUT / "public_locked_validation_cohort_summary_v330.json"]
        files = [p for p in files if p.exists()]

        manifest = {
            "project": "CardioTwin-AI",
            "version": "v3.3.1 Public Locked Inference Pack",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "files_indexed": len(files),
            "files": [
                {
                    "path": p.as_posix(),
                    "size_bytes": int(p.stat().st_size),
                    "sha256": sha256_file(p),
                }
                for p in files
            ],
            "claim_boundary": summary["claim_boundary"],
        }

        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, p.as_posix())
            z.write(manifest_path, manifest_path.as_posix())

        print("ZIP:", zip_path)
        print("MANIFEST:", manifest_path)


if __name__ == "__main__":
    main()
