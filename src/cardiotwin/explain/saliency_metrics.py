from __future__ import annotations
import numpy as np
from cardiotwin.explain.lead_occlusion import lead_occlusion_importance, occlusion_consistency


def saliency_stability_under_noise(model_bundle: dict, signal: np.ndarray, fs: float, leads: list[str],
                                   noise_std: float = 0.02, n_trials: int = 5, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    base = lead_occlusion_importance(model_bundle, signal, fs, leads)["lead_importance_normalized"]
    scores = []
    for _ in range(n_trials):
        noisy = signal + rng.normal(0, noise_std, size=signal.shape).astype(signal.dtype)
        imp = lead_occlusion_importance(model_bundle, noisy, fs, leads)["lead_importance_normalized"]
        scores.append(occlusion_consistency(base, imp))
    return float(np.nanmean(scores)) if scores else float("nan")
