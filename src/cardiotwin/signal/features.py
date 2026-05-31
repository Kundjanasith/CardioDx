from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from cardiotwin.signal.preprocessing import ensure_2d
from cardiotwin.constants import LEADS_12


def detect_r_peaks_proxy(signal: np.ndarray, fs: float, lead_index: int = 1) -> np.ndarray:
    x = ensure_2d(signal)
    y = x[:, min(lead_index, x.shape[1]-1)]
    y = y - np.median(y)
    distance = max(1, int(0.25 * fs))
    prominence = max(0.1, float(np.std(y) * 0.6))
    peaks, _ = find_peaks(y, distance=distance, prominence=prominence)
    return peaks


def extract_global_features(signal: np.ndarray, fs: float) -> dict:
    x = ensure_2d(signal)
    peaks = detect_r_peaks_proxy(x, fs)
    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    hr = 60.0 / np.median(rr) if len(rr) else np.nan
    rr_std = float(np.std(rr)) if len(rr) else np.nan
    rr_cv = float(rr_std / (np.mean(rr) + 1e-9)) if len(rr) else np.nan
    return {
        "heart_rate_proxy": float(hr) if np.isfinite(hr) else -1.0,
        "rr_std_proxy": rr_std if np.isfinite(rr_std) else -1.0,
        "rr_cv_proxy": rr_cv if np.isfinite(rr_cv) else -1.0,
        "r_peak_count": int(len(peaks)),
    }


def extract_features(signal: np.ndarray, fs: float, leads: list[str] | None = None) -> tuple[np.ndarray, list[str], dict]:
    x = ensure_2d(signal)
    if leads is None:
        leads = LEADS_12[:x.shape[1]]
    feats = []
    names = []
    for i, lead in enumerate(leads):
        y = x[:, i]
        vals = {
            "mean": np.mean(y),
            "std": np.std(y),
            "min": np.min(y),
            "max": np.max(y),
            "p01": np.percentile(y, 1),
            "p05": np.percentile(y, 5),
            "p25": np.percentile(y, 25),
            "p50": np.percentile(y, 50),
            "p75": np.percentile(y, 75),
            "p95": np.percentile(y, 95),
            "p99": np.percentile(y, 99),
            "energy": np.mean(y ** 2),
            "abs_mean": np.mean(np.abs(y)),
            "slope_abs_mean": np.mean(np.abs(np.diff(y))),
            "zero_cross": np.mean(np.diff(np.signbit(y)) != 0),
        }
        for k, v in vals.items():
            feats.append(float(v) if np.isfinite(v) else 0.0)
            names.append(f"{lead}_{k}")
    glob = extract_global_features(x, fs)
    for k, v in glob.items():
        feats.append(float(v) if np.isfinite(v) else 0.0)
        names.append(k)
    # simple inter-lead correlations
    for a, b in [("I", "II"), ("V1", "V2"), ("V3", "V4"), ("V5", "V6"), ("II", "aVF")]:
        if a in leads and b in leads:
            ia, ib = leads.index(a), leads.index(b)
            if np.std(x[:, ia]) < 1e-9 or np.std(x[:, ib]) < 1e-9:
                corr = 0.0
            else:
                corr = np.corrcoef(x[:, ia], x[:, ib])[0, 1]
            feats.append(float(corr) if np.isfinite(corr) else 0.0)
            names.append(f"corr_{a}_{b}")
    return np.asarray(feats, dtype=np.float32), names, glob
