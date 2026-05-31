import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_READY = True
except Exception:
    PLOTLY_READY = False

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

st.set_page_config(page_title="CardioTwin-AI v3.0 Real-time ECG Replay", layout="wide")

st.title("CardioTwin-AI v3.0 Real-time ECG Replay")
st.caption("Research-use replay dashboard. This is not live clinical diagnosis.")

def synthetic_ecg(fs=500, seconds=10):
    t = np.arange(0, seconds, 1 / fs)
    data = {}
    for i, lead in enumerate(LEADS):
        base = 0.08 * np.sin(2 * np.pi * 1.2 * t + i * 0.1)
        qrs = np.zeros_like(t)
        for beat in np.arange(0.6, seconds, 0.85):
            qrs += 0.8 * np.exp(-0.5 * ((t - beat) / 0.025) ** 2)
            qrs -= 0.25 * np.exp(-0.5 * ((t - beat + 0.035) / 0.015) ** 2)
        data[lead] = (1 - i * 0.03) * qrs + base + 0.01 * np.random.randn(len(t))
    df = pd.DataFrame(data)
    df.insert(0, "time_sec", t)
    return df

def load_uploaded_csv(file):
    df = pd.read_csv(file)
    missing = [c for c in LEADS if c not in df.columns]
    if missing:
        st.warning(f"Uploaded CSV missing leads: {missing}. Using synthetic demo ECG instead.")
        return synthetic_ecg()
    if "time_sec" not in df.columns:
        df.insert(0, "time_sec", np.arange(len(df)) / 500.0)
    return df[["time_sec"] + LEADS]

def plot_window(df, start_idx, end_idx):
    win = df.iloc[start_idx:end_idx]
    if PLOTLY_READY:
        fig = go.Figure()
        offset = 0
        for lead in LEADS:
            fig.add_trace(go.Scatter(
                x=win["time_sec"],
                y=win[lead] + offset,
                mode="lines",
                name=lead,
                line=dict(width=1),
            ))
            offset += 1.5
        fig.update_layout(
            height=700,
            title="Scrolling 12-lead ECG replay",
            xaxis_title="Time (s)",
            yaxis_title="Lead offset",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(win.set_index("time_sec")[LEADS])

def pseudo_ai_panel(df_window):
    # Placeholder safety demo: replace with CardioTwin inference call when connected.
    amp = float(np.nanmean(np.abs(df_window[LEADS].values)))
    sqi = max(0.0, min(1.0, 1.0 - abs(amp - 0.35)))
    uncertain = sqi < 0.55
    possible_abnormal = amp > 0.45

    return {
        "signal_quality_index": sqi,
        "possible_abnormal_pattern": possible_abnormal,
        "uncertain_or_review_required": uncertain or possible_abnormal,
        "recommendation": "Doctor review / repeat ECG if low SQI" if uncertain else ("Doctor review" if possible_abnormal else "Routine review"),
    }

uploaded = st.sidebar.file_uploader("Optional: upload 12-lead ECG CSV", type=["csv"])
fs = st.sidebar.number_input("Sampling rate (Hz)", min_value=50, max_value=1000, value=500, step=50)
window_sec = st.sidebar.slider("Replay window seconds", 2, 10, 6)
step_sec = st.sidebar.slider("Step seconds", 1, 5, 1)
max_steps = st.sidebar.slider("Max replay steps", 5, 60, 20)
speed = st.sidebar.slider("Replay delay seconds", 0.0, 2.0, 0.3, 0.1)

if uploaded:
    df = load_uploaded_csv(uploaded)
else:
    df = synthetic_ecg(fs=fs, seconds=20)

st.sidebar.write(f"Rows: {len(df)}")
st.sidebar.write(f"Duration: {df['time_sec'].max():.2f} sec")

run = st.sidebar.button("Start replay")

plot_slot = st.empty()
panel_slot = st.empty()

if not run:
    with plot_slot.container():
        plot_window(df, 0, min(len(df), int(window_sec * fs)))
    with panel_slot.container():
        st.info("Press Start replay to simulate real-time ECG streaming.")
else:
    n_window = int(window_sec * fs)
    n_step = int(step_sec * fs)

    for k, start in enumerate(range(0, max(1, len(df) - n_window), n_step)):
        if k >= max_steps:
            break
        end = min(len(df), start + n_window)
        win = df.iloc[start:end]

        with plot_slot.container():
            plot_window(df, start, end)

        result = pseudo_ai_panel(win)
        with panel_slot.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SQI", f"{result['signal_quality_index']:.3f}")
            c2.metric("Possible abnormal", str(result["possible_abnormal_pattern"]))
            c3.metric("Review required", str(result["uncertain_or_review_required"]))
            c4.metric("Recommendation", result["recommendation"])
            st.caption("Pseudo-AI panel for replay demonstration. Connect to CardioTwin inference for production research runs.")

        time.sleep(speed)
