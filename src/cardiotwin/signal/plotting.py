from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cardiotwin.signal.preprocessing import ensure_2d
from cardiotwin.constants import LEADS_12


def plot_12lead(signal: np.ndarray, fs: float, leads: list[str] | None = None, out_path: str | Path | None = None,
                max_seconds: float = 10.0):
    x = ensure_2d(signal)
    leads = leads or LEADS_12[:x.shape[1]]
    n = min(x.shape[0], int(fs * max_seconds))
    t = np.arange(n) / fs
    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        if i >= x.shape[1]:
            ax.axis("off")
            continue
        ax.plot(t, x[:n, i], linewidth=0.7)
        ax.set_title(leads[i])
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    return fig
