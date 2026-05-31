from __future__ import annotations
import numpy as np
from cardiotwin.signal.features import extract_features
from cardiotwin.explain.lead_occlusion import align_feature_vector, bundle_predict_proba

ECG_WINDOWS = {
    "early/P_proxy": (0.00, 0.22),
    "QRS_proxy": (0.22, 0.36),
    "ST_proxy": (0.36, 0.58),
    "T_proxy": (0.58, 0.88),
}

def temporal_occlusion_importance(model_bundle: dict, signal: np.ndarray, fs: float, leads: list[str], fill: str = "mean") -> dict:
    feat, names, _ = extract_features(signal, fs, leads)
    feat = align_feature_vector(feat, names, model_bundle)
    base = bundle_predict_proba(model_bundle, feat.reshape(1, -1))[0]
    labels = model_bundle.get("label_names", [])
    n = signal.shape[0]
    out = {}
    for win, (start_frac, end_frac) in ECG_WINDOWS.items():
        a = int(max(0, min(n, start_frac * n)))
        b = int(max(0, min(n, end_frac * n)))
        x = np.array(signal, copy=True)
        if fill == "mean":
            x[a:b, :] = np.mean(x[a:b, :], axis=0, keepdims=True)
        else:
            x[a:b, :] = 0.0
        f, nm, _ = extract_features(x, fs, leads)
        f = align_feature_vector(f, nm, model_bundle)
        prob = bundle_predict_proba(model_bundle, f.reshape(1, -1))[0]
        delta = np.maximum(base - prob, 0.0)
        out[win] = {labels[i]: float(delta[i]) for i in range(len(labels))}
    return out
