from __future__ import annotations
import numpy as np
from scipy.signal import welch
from cardiotwin.signal.preprocessing import ensure_2d


def _safe_score(value: float, low_bad: float, high_good: float) -> float:
    if value <= low_bad:
        return 0.0
    if value >= high_good:
        return 1.0
    return float((value - low_bad) / (high_good - low_bad))


def compute_lead_sqi(x: np.ndarray, fs: float) -> dict:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x).mean()
    if finite < 0.99:
        return {"sqi": 0.0, "reason": "non_finite"}
    x = np.nan_to_num(x)
    amp = np.percentile(x, 99) - np.percentile(x, 1)
    std = float(np.std(x))
    flat_score = _safe_score(std, 1e-4, 0.01)
    amp_score = 1.0 if 0.02 <= amp <= 10.0 else max(0.0, min(1.0, amp / 0.02 if amp < 0.02 else 10.0 / max(amp, 1e-6)))
    clipping_ratio = float(np.mean(np.abs(x) >= np.percentile(np.abs(x), 99.9))) if len(x) else 1.0
    clip_score = 1.0 if clipping_ratio < 0.01 else max(0.0, 1 - clipping_ratio * 10)
    try:
        f, pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
        total = np.trapz(pxx, f) + 1e-9
        low_power = np.trapz(pxx[f < 0.5], f[f < 0.5]) / total if np.any(f < 0.5) else 0.0
        high_power = np.trapz(pxx[f > 40], f[f > 40]) / total if np.any(f > 40) else 0.0
        baseline_score = max(0.0, 1.0 - low_power * 4)
        hf_score = max(0.0, 1.0 - high_power * 4)
    except Exception:
        baseline_score = 0.5
        hf_score = 0.5
    sqi = float(np.mean([flat_score, amp_score, clip_score, baseline_score, hf_score]))
    warnings = []
    if flat_score < 0.5: warnings.append("flatline_or_too_low_variance")
    if amp_score < 0.5: warnings.append("abnormal_amplitude")
    if baseline_score < 0.5: warnings.append("baseline_wander")
    if hf_score < 0.5: warnings.append("high_frequency_noise")
    return {
        "sqi": sqi,
        "std": std,
        "amplitude_p99_p1": float(amp),
        "baseline_score": float(baseline_score),
        "hf_score": float(hf_score),
        "warnings": warnings,
    }


def compute_sqi(signal: np.ndarray, fs: float, leads: list[str]) -> dict:
    x = ensure_2d(signal)
    lead_scores = {}
    for i, lead in enumerate(leads):
        lead_scores[lead] = compute_lead_sqi(x[:, i], fs)
    overall = float(np.mean([v["sqi"] for v in lead_scores.values()])) if lead_scores else 0.0
    warnings = []
    for lead, info in lead_scores.items():
        for w in info.get("warnings", []):
            warnings.append(f"{lead}:{w}")
    return {"overall_sqi": overall, "lead_sqi": lead_scores, "warnings": warnings}


def is_low_quality(sqi_report: dict, threshold: float = 0.55) -> bool:
    return float(sqi_report.get("overall_sqi", 0.0)) < threshold
