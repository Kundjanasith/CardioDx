from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from cardiotwin.constants import LEADS_12


def synthetic_ecg(fs=500, duration=10, hr=72, noise=0.02, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs*duration))/fs
    sig = np.zeros((len(t), 12), dtype=np.float32)
    period = 60/hr
    r_times = np.arange(0.6, duration, period)
    for li in range(12):
        scale = 1.0 + 0.05*li
        y = np.zeros_like(t)
        for r in r_times:
            y += 0.12*scale*np.exp(-0.5*((t-(r-0.18))/0.035)**2)   # P
            y += -0.15*scale*np.exp(-0.5*((t-(r-0.025))/0.010)**2) # Q
            y += 1.00*scale*np.exp(-0.5*((t-r)/0.012)**2)          # R
            y += -0.25*scale*np.exp(-0.5*((t-(r+0.030))/0.012)**2) # S
            y += 0.35*scale*np.exp(-0.5*((t-(r+0.28))/0.065)**2)   # T
        y += 0.05*np.sin(2*np.pi*0.33*t) + rng.normal(0, noise, len(t))
        if LEADS_12[li] in ["II","III","aVF"]:
            y += 0.03  # small pseudo ST offset for region demo
        sig[:, li] = y
    return sig, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="demo_data")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sig, _ = synthetic_ecg()
    np.savez_compressed(out / "synthetic_12lead_demo.npz", signal=sig, fs=500.0, leads=np.array(LEADS_12), record_id="synthetic_demo")
    # CSV too
    import pandas as pd
    pd.DataFrame(sig, columns=LEADS_12).to_csv(out / "synthetic_12lead_demo.csv", index=False)
    print(f"Saved demo files under {out}")

if __name__ == "__main__":
    main()
