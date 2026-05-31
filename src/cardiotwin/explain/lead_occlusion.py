from __future__ import annotations
import numpy as np
from cardiotwin.signal.features import extract_features
from cardiotwin.models.baseline_ml import predict_proba
from cardiotwin.constants import LEADS_12

_AUGMENTED_LEAD_ALIASES = {
    "aVR": "AVR", "aVL": "AVL", "aVF": "AVF",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
}

def _lead_variants(lead: str) -> list[str]:
    variants = [lead]
    if lead in _AUGMENTED_LEAD_ALIASES:
        variants.append(_AUGMENTED_LEAD_ALIASES[lead])
    return list(dict.fromkeys(variants))

def _feature_name_candidates(name: str) -> list[str]:
    candidates = [name]
    for lead, alias in _AUGMENTED_LEAD_ALIASES.items():
        prefix = lead + "_"
        if name.startswith(prefix):
            candidates.append(alias + "_" + name[len(prefix):])
    if name.startswith("corr_"):
        parts = name[len("corr_"):].split("_")
        if len(parts) == 2:
            a, b = parts
            for va in _lead_variants(a):
                for vb in _lead_variants(b):
                    candidates.append(f"corr_{va}_{vb}")
    return list(dict.fromkeys(candidates))

def align_feature_vector(feat: np.ndarray, names: list[str], model_bundle: dict) -> np.ndarray:
    expected = model_bundle.get("feature_names") or []
    feat = np.asarray(feat, dtype=np.float32)
    if not expected:
        return feat
    feature_map = {name: float(feat[i]) for i, name in enumerate(names)}
    aligned = []
    missing = []
    for expected_name in expected:
        value = None
        for candidate in _feature_name_candidates(expected_name):
            if candidate in feature_map:
                value = feature_map[candidate]
                break
        if value is None:
            missing.append(expected_name)
            value = 0.0
        aligned.append(value)
    return np.asarray(aligned, dtype=np.float32)

def _apply_calibration_if_present(model_bundle: dict, probs: np.ndarray) -> np.ndarray:
    cal = model_bundle.get("calibration") or {}
    calibrators = cal.get("calibrators") or []
    if not calibrators:
        return np.asarray(probs, dtype=np.float32)
    out = np.zeros_like(probs, dtype=np.float32)
    for j in range(probs.shape[1]):
        if j < len(calibrators):
            out[:, j] = calibrators[j].predict(probs[:, j])
        else:
            out[:, j] = probs[:, j]
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def bundle_predict_proba(model_bundle: dict, feat_2d: np.ndarray) -> np.ndarray:
    probs = predict_proba(model_bundle["model"], feat_2d)
    return _apply_calibration_if_present(model_bundle, probs)

def lead_occlusion_importance(
    model_bundle: dict,
    signal: np.ndarray,
    fs: float,
    leads: list[str] | None = None,
    fill: str = "zero",
) -> dict:
    leads = leads or LEADS_12[:signal.shape[1]]
    labels = model_bundle.get("label_names", [])
    feat, names, _ = extract_features(signal, fs, leads)
    feat = align_feature_vector(feat, names, model_bundle)
    base_prob = bundle_predict_proba(model_bundle, feat.reshape(1, -1))[0]

    class_importance = {label: {} for label in labels}
    lead_importance = {lead: 0.0 for lead in leads}
    for i, lead in enumerate(leads):
        x_occ = np.array(signal, copy=True)
        x_occ[:, i] = np.mean(x_occ[:, i]) if fill == "mean" else 0.0
        f_occ, n_occ, _ = extract_features(x_occ, fs, leads)
        f_occ = align_feature_vector(f_occ, n_occ, model_bundle)
        prob = bundle_predict_proba(model_bundle, f_occ.reshape(1, -1))[0]
        delta = np.maximum(base_prob - prob, 0.0)
        lead_importance[lead] = float(np.mean(delta))
        for j, label in enumerate(labels):
            class_importance[label][lead] = float(delta[j])
    total = sum(lead_importance.values()) + 1e-9
    normalized = {k: float(v / total) for k, v in lead_importance.items()}
    return {
        "base_probabilities": {labels[i]: float(base_prob[i]) for i in range(len(labels))},
        "lead_importance": lead_importance,
        "lead_importance_normalized": normalized,
        "class_importance": class_importance,
    }

def occlusion_consistency(importance_a: dict, importance_b: dict) -> float:
    keys = sorted(set(importance_a) & set(importance_b))
    if len(keys) < 2:
        return float("nan")
    a = np.array([importance_a[k] for k in keys], dtype=float)
    b = np.array([importance_b[k] for k in keys], dtype=float)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
