from pathlib import Path

Path("src/cardiotwin/runtime").mkdir(parents=True, exist_ok=True)
Path("src/cardiotwin/runtime/__init__.py").write_text("", encoding="utf-8")

bridge = r'''
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PTBXL_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

LEAD_REGIONS = {
    "septal": ["V1", "V2"],
    "anterior": ["V3", "V4"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "inferior": ["II", "III", "aVF"],
    "global_conduction": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "hypertrophy_chamber": ["I", "aVL", "V5", "V6", "V1", "V2"],
}


def read_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_wfdb_header_fs(header_path: str | Path) -> float:
    header_path = Path(header_path)
    first = header_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
    parts = first.split()
    if len(parts) >= 3:
        try:
            return float(parts[2])
        except Exception:
            return 500.0
    return 500.0


def parse_wfdb_record_name(header_path: str | Path) -> str:
    header_path = Path(header_path)
    first = header_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
    return first.split()[0]


def load_wfdb_hea_mat(header_path: str | Path) -> tuple[np.ndarray, float, dict]:
    """Load WFDB-style .hea + .mat record. Returns signal as shape (12, n_samples)."""
    from scipy.io import loadmat

    header_path = Path(header_path)
    if not header_path.exists():
        raise FileNotFoundError(f"Header not found: {header_path}")

    fs = parse_wfdb_header_fs(header_path)
    record_name = parse_wfdb_record_name(header_path)
    mat_path = header_path.with_suffix(".mat")

    if not mat_path.exists():
        # Some headers may reference a different mat filename.
        for line in header_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ".mat" in line:
                candidate = header_path.parent / line.split()[0]
                if candidate.exists():
                    mat_path = candidate
                    break

    if not mat_path.exists():
        raise FileNotFoundError(f"MAT pair not found for: {header_path}")

    mat = loadmat(mat_path)
    if "val" in mat:
        x = np.asarray(mat["val"], dtype=np.float32)
    else:
        candidates = [v for v in mat.values() if isinstance(v, np.ndarray) and v.ndim == 2]
        if not candidates:
            raise RuntimeError(f"No 2D waveform array found inside {mat_path}")
        x = np.asarray(candidates[0], dtype=np.float32)

    # Ensure (12, n)
    if x.shape[0] != 12 and x.shape[1] == 12:
        x = x.T

    if x.shape[0] != 12:
        raise RuntimeError(f"Expected 12 leads, got shape={x.shape}")

    meta = {
        "record_name": record_name,
        "header_path": str(header_path),
        "mat_path": str(mat_path),
        "raw_fs": fs,
        "raw_shape": list(x.shape),
    }
    return x, fs, meta


def load_csv_12lead(csv_file_or_path: Any, fs: float = 500.0) -> tuple[np.ndarray, float, dict]:
    df = pd.read_csv(csv_file_or_path)
    missing = [c for c in LEADS if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required ECG leads: {missing}")

    x = df[LEADS].to_numpy(dtype=np.float32).T
    meta = {
        "record_name": "uploaded_csv",
        "raw_fs": fs,
        "raw_shape": list(x.shape),
        "columns": list(df.columns),
    }
    return x, fs, meta


def synthetic_ecg(fs: float = 500.0, seconds: float = 10.0, pattern: str = "balanced") -> tuple[np.ndarray, float, dict]:
    t = np.arange(0, seconds, 1 / fs)
    data = []

    for i, lead in enumerate(LEADS):
        p = np.zeros_like(t)
        qrs = np.zeros_like(t)
        tw = np.zeros_like(t)

        for beat in np.arange(0.65, seconds, 0.86):
            p += 0.06 * np.exp(-0.5 * ((t - (beat - 0.18)) / 0.035) ** 2)
            qrs += 0.90 * np.exp(-0.5 * ((t - beat) / 0.022) ** 2)
            qrs -= 0.24 * np.exp(-0.5 * ((t - (beat - 0.026)) / 0.014) ** 2)
            qrs -= 0.18 * np.exp(-0.5 * ((t - (beat + 0.035)) / 0.016) ** 2)
            tw += 0.20 * np.exp(-0.5 * ((t - (beat + 0.27)) / 0.085) ** 2)

        baseline = 0.04 * np.sin(2 * np.pi * 0.30 * t + i * 0.12)
        sig = (1.0 - i * 0.018) * (p + qrs + tw) + baseline

        if pattern == "inferior_mi_like" and lead in ["II", "III", "aVF"]:
            sig += 0.16 * np.exp(-0.5 * ((np.mod(t, 0.86) - 0.43) / 0.09) ** 2)
        elif pattern == "anterior_sttc_like" and lead in ["V3", "V4", "V5"]:
            sig += 0.10 * np.sin(2 * np.pi * 1.2 * t)
        elif pattern == "lateral_voltage_like" and lead in ["I", "aVL", "V5", "V6"]:
            sig *= 1.25
        elif pattern == "low_quality":
            sig += 0.15 * np.random.randn(len(t))

        sig += 0.006 * np.random.randn(len(t))
        data.append(sig.astype(np.float32))

    x = np.vstack(data)
    meta = {
        "record_name": f"synthetic_{pattern}",
        "raw_fs": fs,
        "raw_shape": list(x.shape),
        "pattern": pattern,
    }
    return x, fs, meta


def resample_to_target(x: np.ndarray, fs: float, target_fs: float = 100.0, seconds: float = 10.0) -> np.ndarray:
    """Return shape (12, target_len)."""
    from scipy.signal import resample

    x = np.asarray(x, dtype=np.float32)
    target_len = int(round(target_fs * seconds))

    # Crop or pad raw to desired seconds first.
    raw_len = int(round(fs * seconds))
    if x.shape[1] >= raw_len:
        x = x[:, :raw_len]
    else:
        pad = raw_len - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad)), mode="constant")

    if abs(fs - target_fs) < 1e-6:
        y = x
    else:
        y = resample(x, target_len, axis=1).astype(np.float32)

    # Robust per-lead normalization
    med = np.nanmedian(y, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(y - med), axis=1, keepdims=True)
    scale = np.where(mad > 1e-6, 1.4826 * mad, np.nanstd(y, axis=1, keepdims=True) + 1e-6)
    y = (y - med) / scale
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return y[:, :target_len]


def estimate_sqi(x_12n: np.ndarray) -> float:
    x = np.asarray(x_12n, dtype=float)
    finite = np.isfinite(x).mean()
    amp = float(np.nanmedian(np.abs(x)))
    noise = float(np.nanmedian(np.abs(np.diff(x, axis=1)))) if x.shape[1] > 2 else 0.0
    flat = float((np.nanstd(x, axis=1) < 1e-6).mean())

    amp_score = np.clip(amp / 0.75, 0, 1)
    noise_score = 1 - np.clip(noise / 0.60, 0, 1)
    sqi = 0.48 * amp_score + 0.37 * noise_score + 0.15 * finite
    sqi *= 1 - 0.45 * flat
    return float(np.clip(sqi, 0, 1))


def lead_amplitudes(x_12n: np.ndarray) -> dict:
    return {lead: float(np.nanmean(np.abs(x_12n[i]))) for i, lead in enumerate(LEADS)}


def fallback_region_mapper(x_12n: np.ndarray, probabilities: dict | None = None) -> dict:
    amp = lead_amplitudes(x_12n)
    mean_amp = np.mean(list(amp.values())) + 1e-9
    scores = {}

    for region, leads in LEAD_REGIONS.items():
        vals = [amp[l] for l in leads]
        rel = float(np.mean(vals) / mean_amp)
        scores[region] = float(np.clip((rel - 0.75) / 0.85, 0, 1))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    second, second_score = ranked[1]
    margin = top_score - second_score

    if top_score < 0.18:
        decision = "uncertain"
        reason = "low_region_evidence"
    elif margin < 0.08:
        decision = "uncertain"
        reason = "top_region_margin_too_small"
    else:
        decision = top
        reason = "dominant_region_evidence"

    return {
        "source": "fallback_lead_region_mapper",
        "scores": scores,
        "lead_amplitudes": amp,
        "decision": decision,
        "reason": reason,
        "top_region": top,
        "top_score": float(top_score),
        "second_region": second,
        "second_score": float(second_score),
        "margin": float(margin),
    }


def extract_threshold_number(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return float(default)
    if isinstance(value, dict):
        for key in [
            "threshold", "value", "tuned_threshold", "selected_threshold",
            "operating_threshold", "cutoff", "cut_off", "decision_threshold",
        ]:
            if key in value:
                return extract_threshold_number(value[key], default)
        for key in ["screening", "balanced", "safety", "default"]:
            if key in value:
                return extract_threshold_number(value[key], default)
    return float(default)


def threshold_dict_from_obj(obj: Any, defaults: dict) -> dict | None:
    if not isinstance(obj, dict):
        return None

    if any(k in obj for k in defaults):
        return {k: extract_threshold_number(obj.get(k, defaults[k]), defaults[k]) for k in defaults}

    for key in [
        "thresholds", "class_thresholds", "per_class_thresholds",
        "per_class", "labels", "classes",
    ]:
        if key in obj and isinstance(obj[key], dict):
            found = threshold_dict_from_obj(obj[key], defaults)
            if found is not None:
                return found
    return None


def load_thresholds(path: str | Path, profile: str = "screening") -> tuple[dict, str]:
    defaults = {"NORM": 0.13, "MI": 0.50, "STTC": 0.15, "CD": 0.13, "HYP": 0.10}
    raw = read_json(path)
    if not raw:
        return defaults, "fallback_defaults"

    found = threshold_dict_from_obj(raw, defaults)
    if found is not None:
        return found, "thresholds:root"

    for key in [profile, "screening", "balanced", "safety", "default", "profiles", "threshold_profiles", "operating_profiles", "thresholds", "deep_thresholds"]:
        if not isinstance(raw, dict) or key not in raw:
            continue
        obj = raw[key]

        if isinstance(obj, dict) and profile in obj:
            found = threshold_dict_from_obj(obj[profile], defaults)
            if found is not None:
                return found, f"thresholds:{key}.{profile}"

        found = threshold_dict_from_obj(obj, defaults)
        if found is not None:
            return found, f"thresholds:{key}"

    if isinstance(raw, dict):
        for key, obj in raw.items():
            found = threshold_dict_from_obj(obj, defaults)
            if found is not None:
                return found, f"thresholds:auto_scan:{key}"

    return defaults, "threshold_file_unparsed_fallback"


def _try_import_v27():
    info = {"available": {}, "errors": {}}

    try:
        from cardiotwin.signal.preprocessing import preprocess_ecg, pad_or_crop
        info["available"]["preprocessing"] = True
    except Exception as e:
        preprocess_ecg = None
        pad_or_crop = None
        info["available"]["preprocessing"] = False
        info["errors"]["preprocessing"] = repr(e)

    try:
        from cardiotwin.models.deep_ecg import make_deep_model
        info["available"]["make_deep_model"] = True
    except Exception as e:
        make_deep_model = None
        info["available"]["make_deep_model"] = False
        info["errors"]["make_deep_model"] = repr(e)

    try:
        from cardiotwin.explain.region_mapper_v23 import map_prediction_to_region
        info["available"]["region_mapper_v23"] = True
    except Exception as e:
        map_prediction_to_region = None
        info["available"]["region_mapper_v23"] = False
        info["errors"]["region_mapper_v23"] = repr(e)

    return preprocess_ecg, pad_or_crop, make_deep_model, map_prediction_to_region, info


def make_model_flexible(make_deep_model, n_classes: int = 5):
    if make_deep_model is None:
        return None, "make_deep_model_unavailable"

    attempts = [
        {"n_classes": n_classes},
        {"num_classes": n_classes},
        {"out_dim": n_classes},
        {"in_channels": 12, "n_classes": n_classes},
        {"in_chans": 12, "num_classes": n_classes},
        {},
    ]

    errors = []
    for kwargs in attempts:
        try:
            model = make_deep_model(**kwargs)
            return model, f"make_deep_model({kwargs})"
        except Exception as e:
            errors.append(f"{kwargs}: {repr(e)}")

    return None, "make_deep_model_failed: " + " | ".join(errors[:3])


def load_torch_model(model_path: str | Path, device: str = "cpu") -> tuple[Any, dict]:
    import torch

    model_path = Path(model_path)
    meta = {
        "model_path": str(model_path),
        "device": device,
        "loaded": False,
        "mode": None,
        "errors": [],
    }

    if not model_path.exists():
        meta["errors"].append("model_path_missing")
        return None, meta

    preprocess_ecg, pad_or_crop, make_deep_model, _, import_info = _try_import_v27()
    meta["import_info"] = import_info

    obj = torch.load(model_path, map_location=device)

    # Case 1: whole nn.Module saved.
    if hasattr(obj, "eval") and callable(obj.eval):
        obj.eval()
        meta["loaded"] = True
        meta["mode"] = "torch_module_direct"
        return obj, meta

    # Case 2: checkpoint dict.
    state_dict = None
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in obj:
                candidate = obj[key]
                if isinstance(candidate, dict):
                    state_dict = candidate
                    break

        if state_dict is None and all(hasattr(v, "shape") for v in obj.values() if hasattr(v, "shape")):
            state_dict = obj

    model, make_mode = make_model_flexible(make_deep_model, n_classes=len(PTBXL_LABELS))
    meta["make_mode"] = make_mode

    if model is None:
        meta["errors"].append("could_not_construct_model")
        return None, meta

    if state_dict is None:
        meta["errors"].append("no_state_dict_found")
        return None, meta

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()
        meta["loaded"] = True
        meta["mode"] = "constructed_model_state_dict"
        meta["missing_keys_count"] = len(list(missing))
        meta["unexpected_keys_count"] = len(list(unexpected))
        return model, meta
    except Exception as e:
        meta["errors"].append("load_state_dict_failed: " + repr(e))
        return None, meta


def infer_with_model(model: Any, x_ai: np.ndarray, device: str = "cpu") -> tuple[dict, dict]:
    import torch

    meta = {"input_shape_tried": [], "selected_input_shape": None, "raw_output_shape": None}

    candidates = [
        x_ai[None, :, :],          # (1, 12, 1000)
        x_ai.T[None, :, :],        # (1, 1000, 12)
    ]

    last_error = None

    with torch.no_grad():
        for arr in candidates:
            try:
                tensor = torch.tensor(arr, dtype=torch.float32, device=device)
                meta["input_shape_tried"].append(list(tensor.shape))
                out = model(tensor)

                if isinstance(out, (tuple, list)):
                    out = out[0]

                out_np = out.detach().cpu().numpy()
                meta["raw_output_shape"] = list(out_np.shape)
                meta["selected_input_shape"] = list(tensor.shape)

                out_np = np.asarray(out_np).reshape(-1)

                # Use sigmoid for multi-label logits. If already probability-like, sigmoid still
                # can distort, so detect range first.
                if np.nanmin(out_np) >= 0 and np.nanmax(out_np) <= 1:
                    probs = out_np
                    meta["activation"] = "identity_probability_like"
                else:
                    probs = 1 / (1 + np.exp(-out_np))
                    meta["activation"] = "sigmoid_logits"

                if len(probs) < len(PTBXL_LABELS):
                    raise RuntimeError(f"Model output too short: {len(probs)}")

                probs = probs[:len(PTBXL_LABELS)]
                return {label: float(probs[i]) for i, label in enumerate(PTBXL_LABELS)}, meta

            except Exception as e:
                last_error = repr(e)

    raise RuntimeError(f"Model inference failed for all input layouts: {last_error}")


def try_region_mapper_v23(map_prediction_to_region, x_ai: np.ndarray, probabilities: dict, positive_labels: list[str]) -> tuple[list[dict], dict]:
    meta = {"used": False, "errors": []}

    if map_prediction_to_region is None:
        meta["errors"].append("region_mapper_v23_import_unavailable")
        return [], meta

    decisions = []

    for label in positive_labels:
        prob = probabilities.get(label, 0.0)

        call_attempts = [
            lambda: map_prediction_to_region(class_name=label, class_probability=prob, ecg=x_ai, leads=LEADS),
            lambda: map_prediction_to_region(label, prob, x_ai),
            lambda: map_prediction_to_region(label, prob),
            lambda: map_prediction_to_region({"class": label, "probability": prob, "ecg": x_ai, "leads": LEADS}),
        ]

        for fn in call_attempts:
            try:
                res = fn()
                if isinstance(res, dict):
                    d = dict(res)
                else:
                    d = {"raw_result": str(res)}
                d["class"] = label
                d["class_probability"] = float(prob)
                decisions.append(d)
                meta["used"] = True
                break
            except Exception as e:
                meta["errors"].append(f"{label}: {repr(e)}")

    return decisions, meta


def run_v304_real_inference(
    x_raw: np.ndarray,
    fs: float,
    model_path: str | Path = "artifacts/models/inceptiontime_v21_safety.pt",
    threshold_path: str | Path = "artifacts/deep_safety_v21/threshold_profiles_deep.json",
    profile: str = "screening",
    device: str = "cpu",
    source_meta: dict | None = None,
) -> dict:
    preprocess_ecg, pad_or_crop, make_deep_model, map_prediction_to_region, import_info = _try_import_v27()

    x_ai = resample_to_target(x_raw, fs=fs, target_fs=100.0, seconds=10.0)
    sqi = estimate_sqi(x_ai)

    thresholds, threshold_source = load_thresholds(threshold_path, profile=profile)

    model, model_meta = load_torch_model(model_path, device=device)

    if model is not None and model_meta.get("loaded"):
        try:
            probabilities, inference_meta = infer_with_model(model, x_ai, device=device)
            inference_mode = "real_v2_7_torch_model"
            inference_error = None
        except Exception as e:
            probabilities = None
            inference_meta = {"error": repr(e)}
            inference_mode = "real_model_loaded_but_inference_failed"
            inference_error = repr(e)
    else:
        probabilities = None
        inference_meta = {"error": "model_not_loaded"}
        inference_mode = "model_unavailable"
        inference_error = "; ".join(model_meta.get("errors", []))

    # Safe fallback if real model cannot be called.
    if probabilities is None:
        region_hint = fallback_region_mapper(x_ai)
        s = region_hint["scores"]
        probabilities = {
            "NORM": float(np.clip(0.74 * sqi - 0.18 * max(s.values()), 0.02, 0.98)),
            "MI": float(np.clip(0.08 + 0.48 * s["inferior"] + 0.30 * s["anterior"], 0.01, 0.96)),
            "STTC": float(np.clip(0.10 + 0.56 * s["anterior"], 0.01, 0.96)),
            "CD": float(np.clip(0.08 + 0.42 * s["global_conduction"] + 0.14 * (1 - sqi), 0.01, 0.96)),
            "HYP": float(np.clip(0.08 + 0.50 * s["hypertrophy_chamber"] + 0.08 * s["lateral"], 0.01, 0.96)),
        }
        inference_mode = "fallback_demo_inference_due_to_model_bridge_failure"

    positive_labels = [k for k in PTBXL_LABELS if probabilities.get(k, 0.0) >= thresholds.get(k, 0.5)]
    abnormal_positive_labels = [k for k in positive_labels if k != "NORM"]

    # Region mapper v2.3 if possible, fallback otherwise.
    region_decisions, region_meta = try_region_mapper_v23(
        map_prediction_to_region=map_prediction_to_region,
        x_ai=x_ai,
        probabilities=probabilities,
        positive_labels=abnormal_positive_labels,
    )

    fallback_region = fallback_region_mapper(x_ai, probabilities=probabilities)

    if region_decisions:
        region_summary = {
            "source": "region_mapper_v23",
            "decisions": region_decisions,
            "fallback_region": fallback_region,
        }
    else:
        region_summary = fallback_region

    low_sqi = sqi < 0.55
    uncertain = low_sqi or (not abnormal_positive_labels and probabilities.get("NORM", 0) < thresholds.get("NORM", 0.5))

    if low_sqi:
        recommendation = "Repeat ECG / doctor review due to low signal quality"
    elif "MI" in abnormal_positive_labels:
        recommendation = "Urgent doctor review for possible MI-like screening pattern"
    elif abnormal_positive_labels:
        recommendation = "Doctor review recommended"
    elif uncertain:
        recommendation = "Doctor review recommended due to uncertainty"
    else:
        recommendation = "Routine review"

    return {
        "version": "CardioTwin-AI v3.0.4 real inference bridge",
        "source_meta": source_meta or {},
        "raw_fs": float(fs),
        "ai_fs": 100.0,
        "raw_shape": list(np.asarray(x_raw).shape),
        "ai_shape": list(x_ai.shape),
        "labels": PTBXL_LABELS,
        "sqi": float(sqi),
        "model_path": str(model_path),
        "threshold_path": str(threshold_path),
        "threshold_profile": profile,
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "probabilities": probabilities,
        "positive_labels": positive_labels,
        "abnormal_positive_labels": abnormal_positive_labels,
        "low_sqi": low_sqi,
        "uncertain": uncertain,
        "recommendation": recommendation,
        "inference_mode": inference_mode,
        "inference_error": inference_error,
        "model_meta": model_meta,
        "inference_meta": inference_meta,
        "import_info": import_info,
        "region_mapper_meta": region_meta,
        "region_summary": region_summary,
        "claim_boundary": "Research-use preliminary screening support. Not final diagnosis.",
    }
'''

Path("src/cardiotwin/runtime/v304_real_inference_bridge.py").write_text(bridge, encoding="utf-8")
print("DONE: src/cardiotwin/runtime/v304_real_inference_bridge.py")
