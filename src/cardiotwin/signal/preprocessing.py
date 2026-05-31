from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from math import gcd


def ensure_2d(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < x.shape[1] and x.shape[0] <= 12:
        x = x.T
    return x


def resample_signal(signal: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    x = ensure_2d(signal)
    if abs(fs - target_fs) < 1e-6:
        return x.astype(np.float32)
    g = gcd(int(round(fs)), int(round(target_fs)))
    up = int(round(target_fs)) // g
    down = int(round(fs)) // g
    return resample_poly(x, up, down, axis=0).astype(np.float32)


def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 3) -> np.ndarray:
    x = ensure_2d(signal)
    nyq = fs / 2.0
    high = min(high, nyq - 1e-3)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    if x.shape[0] <= max(len(a), len(b)) * 3:
        return x.astype(np.float32)
    return filtfilt(b, a, x, axis=0).astype(np.float32)


def notch_filter(signal: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    x = ensure_2d(signal)
    nyq = fs / 2.0
    if freq >= nyq:
        return x.astype(np.float32)
    b, a = iirnotch(w0=freq / nyq, Q=q)
    if x.shape[0] <= max(len(a), len(b)) * 3:
        return x.astype(np.float32)
    return filtfilt(b, a, x, axis=0).astype(np.float32)


def robust_normalize(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = ensure_2d(signal)
    med = np.median(x, axis=0, keepdims=True)
    iqr = np.percentile(x, 75, axis=0, keepdims=True) - np.percentile(x, 25, axis=0, keepdims=True)
    return ((x - med) / (iqr + eps)).astype(np.float32)


def preprocess_ecg(signal: np.ndarray, fs: float, target_fs: float = 100.0, notch: float | None = 50.0,
                   normalize: bool = True) -> tuple[np.ndarray, float]:
    x = ensure_2d(signal)
    x = resample_signal(x, fs, target_fs)
    fs2 = float(target_fs)
    x = bandpass_filter(x, fs2, low=0.5, high=min(40.0, fs2/2 - 1.0))
    if notch is not None:
        x = notch_filter(x, fs2, freq=notch)
    if normalize:
        x = robust_normalize(x)
    return x.astype(np.float32), fs2


def pad_or_crop(signal: np.ndarray, n_samples: int) -> np.ndarray:
    x = ensure_2d(signal)
    if x.shape[0] == n_samples:
        return x.astype(np.float32)
    if x.shape[0] > n_samples:
        start = (x.shape[0] - n_samples) // 2
        return x[start:start+n_samples].astype(np.float32)
    pad = n_samples - x.shape[0]
    left = pad // 2
    right = pad - left
    return np.pad(x, ((left, right), (0, 0)), mode="edge").astype(np.float32)
