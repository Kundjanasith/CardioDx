from __future__ import annotations
import numpy as np
from cardiotwin.constants import LEADS_12
from cardiotwin.signal.preprocessing import preprocess_ecg, pad_or_crop
from cardiotwin.signal.quality import compute_sqi
from cardiotwin.signal.features import extract_features
from cardiotwin.explain.lead_occlusion import lead_occlusion_importance, align_feature_vector, bundle_predict_proba
from cardiotwin.mapping.digital_twin_state import build_digital_twin_state
from cardiotwin.eval.safety_v12 import safety_decisions, threshold_predict

def run_inference(
    model_bundle: dict,
    signal: np.ndarray,
    fs: float,
    leads: list[str] | None = None,
    record_id: str = "uploaded",
    target_fs: float = 100.0,
    duration_sec: float = 10.0,
    threshold_profile: str = "balanced",
) -> dict:
    leads = leads or LEADS_12[:signal.shape[1]]
    x, fs2 = preprocess_ecg(signal, fs, target_fs=target_fs, normalize=True)
    x = pad_or_crop(x, int(target_fs * duration_sec))
    sqi = compute_sqi(x, fs2, leads)
    feat, names, glob = extract_features(x, fs2, leads)
    feat_aligned = align_feature_vector(feat, names, model_bundle)
    probs = bundle_predict_proba(model_bundle, feat_aligned.reshape(1, -1))[0]
    labels = model_bundle.get("label_names", [])
    class_probs = {labels[i]: float(probs[i]) for i in range(len(labels))}

    thresholds = model_bundle.get("threshold_profiles") or model_bundle.get("thresholds") or {}
    if thresholds:
        y_pred = threshold_predict(probs.reshape(1, -1), thresholds, labels, profile=threshold_profile)[0]
        predicted_labels = [labels[i] for i, v in enumerate(y_pred) if int(v) == 1]
    else:
        y_pred = (probs >= 0.5).astype(int)
        predicted_labels = [labels[i] for i, v in enumerate(y_pred) if int(v) == 1]

    decisions = safety_decisions(
        probs.reshape(1, -1),
        np.array([sqi.get("overall_sqi", 0.0)]),
        thresholds if thresholds else {label: 0.5 for label in labels},
        labels,
        profile=threshold_profile,
    )
    safety = decisions[0]

    occ = lead_occlusion_importance(model_bundle, x, fs2, leads)
    state = build_digital_twin_state(record_id, class_probs, occ["lead_importance_normalized"], sqi)
    state["threshold_profile"] = threshold_profile
    state["thresholded_prediction"] = {label: int(label in predicted_labels) for label in labels}
    state["predicted_labels"] = predicted_labels
    state["safety_gate"] = safety
    state["clinical_interpretation_status"] = safety["status"]
    state["preprocessing"] = {
        "target_fs": fs2,
        "duration_sec": duration_sec,
        "raw_feature_count": int(len(feat)),
        "model_feature_count": int(len(feat_aligned)),
        "feature_alignment": "enabled",
    }
    state["global_features"] = glob
    state["explainability"] = occ
    return state
