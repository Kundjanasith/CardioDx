from pathlib import Path

app = r'''
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_READY = True
except Exception:
    PLOTLY_READY = False


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

LEAD_REGIONS = {
    "septal": ["V1", "V2"],
    "anterior": ["V3", "V4"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "inferior": ["II", "III", "aVF"],
    "global_conduction": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "hypertrophy_chamber": ["I", "aVL", "V5", "V6", "V1", "V2"],
}

REGION_COORDS = {
    "septal": (0.0, 0.0, 0.35),
    "anterior": (0.0, 0.7, 0.25),
    "lateral": (0.85, 0.1, 0.1),
    "inferior": (0.0, -0.9, -0.1),
    "global_conduction": (-0.75, 0.25, 0.25),
    "hypertrophy_chamber": (0.55, -0.15, -0.25),
}


st.set_page_config(
    page_title="CardioTwin-AI v3.0.1 Replay + 3D/4D Heart",
    layout="wide",
)


def synthetic_ecg(fs=500, seconds=20, abnormal_mode="mild"):
    t = np.arange(0, seconds, 1 / fs)
    data = {}

    for i, lead in enumerate(LEADS):
        base = 0.05 * np.sin(2 * np.pi * 0.33 * t + i * 0.07)
        p_wave = np.zeros_like(t)
        qrs = np.zeros_like(t)
        t_wave = np.zeros_like(t)

        for beat in np.arange(0.6, seconds, 0.85):
            p_wave += 0.08 * np.exp(-0.5 * ((t - (beat - 0.18)) / 0.035) ** 2)
            qrs += 0.95 * np.exp(-0.5 * ((t - beat) / 0.020) ** 2)
            qrs -= 0.28 * np.exp(-0.5 * ((t - (beat - 0.025)) / 0.012) ** 2)
            qrs -= 0.20 * np.exp(-0.5 * ((t - (beat + 0.035)) / 0.018) ** 2)
            t_wave += 0.22 * np.exp(-0.5 * ((t - (beat + 0.26)) / 0.080) ** 2)

        lead_gain = 1.0 - i * 0.025
        signal = lead_gain * (p_wave + qrs + t_wave) + base

        if abnormal_mode == "sttc" and lead in ["V3", "V4", "V5"]:
            signal += 0.10 * np.sin(2 * np.pi * 1.1 * t)
        elif abnormal_mode == "inferior" and lead in ["II", "III", "aVF"]:
            signal += 0.18 * np.exp(-0.5 * ((np.mod(t, 0.85) - 0.42) / 0.10) ** 2)
        elif abnormal_mode == "lateral" and lead in ["I", "aVL", "V5", "V6"]:
            signal *= 1.18

        signal += 0.008 * np.random.randn(len(t))
        data[lead] = signal

    df = pd.DataFrame(data)
    df.insert(0, "time_sec", t)
    return df


def load_uploaded_csv(file, fs):
    df = pd.read_csv(file)
    missing = [c for c in LEADS if c not in df.columns]
    if missing:
        st.warning(f"Uploaded CSV missing leads: {missing}. Using synthetic demo ECG instead.")
        return synthetic_ecg(fs=fs, seconds=20)

    if "time_sec" not in df.columns:
        df.insert(0, "time_sec", np.arange(len(df)) / fs)

    return df[["time_sec"] + LEADS]


def phase_name(t):
    cycle = 0.85
    ph = (t % cycle) / cycle

    if ph < 0.18:
        return "P wave / atrial activation", ph
    if ph < 0.32:
        return "QRS / ventricular depolarization", ph
    if ph < 0.55:
        return "ST segment", ph
    if ph < 0.78:
        return "T wave / repolarization", ph
    return "TP baseline", ph


def estimate_sqi(df_window):
    x = df_window[LEADS].values.astype(float)

    amp = float(np.nanmedian(np.abs(x)))
    noise = float(np.nanmedian(np.abs(np.diff(x, axis=0)))) if len(x) > 2 else 0.0
    finite_ratio = float(np.isfinite(x).mean())
    flat_ratio = float((np.nanstd(x, axis=0) < 1e-4).mean())

    amp_score = np.clip(amp / 0.35, 0, 1)
    noise_penalty = np.clip(noise / 0.08, 0, 1)
    flat_penalty = np.clip(flat_ratio, 0, 1)

    sqi = 0.55 * amp_score + 0.35 * (1 - noise_penalty) + 0.10 * finite_ratio
    sqi = sqi * (1 - 0.50 * flat_penalty)
    return float(np.clip(sqi, 0, 1))


def lead_amplitudes(df_window):
    vals = {}
    for lead in LEADS:
        x = df_window[lead].values.astype(float)
        vals[lead] = float(np.nanmean(np.abs(x)))
    return vals


def region_evidence(df_window):
    lead_amp = lead_amplitudes(df_window)
    scores = {}

    for region, leads in LEAD_REGIONS.items():
        raw = np.mean([lead_amp.get(l, 0.0) for l in leads])
        norm = raw / (np.mean(list(lead_amp.values())) + 1e-9)
        scores[region] = float(np.clip((norm - 0.75) / 0.85, 0, 1))

    sorted_regions = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_region, top_score = sorted_regions[0]
    second_region, second_score = sorted_regions[1]
    margin = top_score - second_score

    if top_score < 0.18:
        decision = "uncertain"
        reason = "low_region_evidence"
    elif margin < 0.08:
        decision = "uncertain"
        reason = "top_region_margin_too_small"
    else:
        decision = top_region
        reason = "dominant_region_evidence"

    return {
        "scores": scores,
        "top_region": top_region,
        "top_score": top_score,
        "second_region": second_region,
        "second_score": second_score,
        "margin": margin,
        "decision": decision,
        "reason": reason,
        "lead_amplitudes": lead_amp,
    }


def pseudo_ai_panel(df_window):
    sqi = estimate_sqi(df_window)
    amp = lead_amplitudes(df_window)
    ev = region_evidence(df_window)

    lateral = np.mean([amp[l] for l in ["I", "aVL", "V5", "V6"]])
    inferior = np.mean([amp[l] for l in ["II", "III", "aVF"]])
    septal = np.mean([amp[l] for l in ["V1", "V2"]])
    anterior = np.mean([amp[l] for l in ["V3", "V4"]])
    global_amp = np.mean(list(amp.values()))

    probs = {
        "NORM": float(np.clip(0.78 * sqi - 0.25 * max(ev["scores"].values()), 0.02, 0.98)),
        "MI": float(np.clip(0.10 + 0.45 * ev["scores"]["inferior"] + 0.30 * ev["scores"]["anterior"], 0.01, 0.95)),
        "STTC": float(np.clip(0.12 + 0.55 * ev["scores"]["anterior"], 0.01, 0.95)),
        "CD": float(np.clip(0.08 + 0.45 * ev["scores"]["global_conduction"] + 0.15 * (1 - sqi), 0.01, 0.95)),
        "HYP": float(np.clip(0.08 + 0.50 * ev["scores"]["hypertrophy_chamber"] + 0.10 * lateral / (global_amp + 1e-9), 0.01, 0.95)),
    }

    thresholds = {
        "NORM": 0.50,
        "MI": 0.30,
        "STTC": 0.30,
        "CD": 0.25,
        "HYP": 0.30,
    }

    positives = [k for k, v in probs.items() if v >= thresholds[k]]
    abnormal_positive = [k for k in positives if k != "NORM"]
    low_sqi = sqi < 0.55
    uncertain = low_sqi or (not abnormal_positive and probs["NORM"] < 0.50)

    if low_sqi:
        recommendation = "Repeat ECG / doctor review due to low SQI"
    elif "MI" in abnormal_positive:
        recommendation = "Urgent doctor review for possible MI-like pattern"
    elif abnormal_positive:
        recommendation = "Doctor review recommended"
    else:
        recommendation = "Routine review"

    return {
        "sqi": sqi,
        "probabilities": probs,
        "thresholds": thresholds,
        "positive_labels": positives,
        "abnormal_positive_labels": abnormal_positive,
        "uncertain": uncertain,
        "region": ev,
        "recommendation": recommendation,
    }


def plot_ecg_window(df, start_idx, end_idx):
    win = df.iloc[start_idx:end_idx]

    if not PLOTLY_READY:
        st.line_chart(win.set_index("time_sec")[LEADS])
        return None

    fig = go.Figure()
    offset = 0.0

    for lead in LEADS:
        fig.add_trace(
            go.Scatter(
                x=win["time_sec"],
                y=win[lead] + offset,
                mode="lines",
                name=lead,
                line=dict(width=1),
            )
        )
        offset += 1.35

    fig.update_layout(
        height=620,
        title="Live 12-lead ECG replay",
        xaxis_title="Time (s)",
        yaxis_title="Lead offset",
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)
    return fig


def heart_surface(region_scores, phase):
    u = np.linspace(0, 2 * np.pi, 95)
    v = np.linspace(-1.0, 1.0, 28)
    U, V = np.meshgrid(u, v)

    X2 = 16 * np.sin(U) ** 3
    Y2 = 13 * np.cos(U) - 5 * np.cos(2 * U) - 2 * np.cos(3 * U) - np.cos(4 * U)

    X = X2 / 18.0
    Y = Y2 / 18.0
    Z = V * (0.55 + 0.15 * np.cos(U))

    # Electrical wave front sweeps from top to bottom.
    wave_front = 0.80 - 1.70 * phase
    wave = np.exp(-((Y - wave_front) ** 2) / 0.035)

    color = 0.10 + 0.75 * wave

    color += region_scores.get("anterior", 0) * np.exp(-((Y - 0.55) ** 2 + X ** 2) / 0.20)
    color += region_scores.get("inferior", 0) * np.exp(-((Y + 0.70) ** 2 + X ** 2) / 0.20)
    color += region_scores.get("lateral", 0) * np.exp(-((np.abs(X) - 0.65) ** 2 + (Y - 0.05) ** 2) / 0.20)
    color += region_scores.get("septal", 0) * np.exp(-(X ** 2 + (Y - 0.05) ** 2) / 0.12)
    color += region_scores.get("hypertrophy_chamber", 0) * np.exp(-((Y + 0.05) ** 2 + (X - 0.35) ** 2) / 0.25)
    color += region_scores.get("global_conduction", 0) * 0.15

    color = np.clip(color, 0, 1.5)
    return X, Y, Z, color


def plot_heart_map(region_info, phase, phase_label):
    if not PLOTLY_READY:
        st.info("Plotly is not available.")
        return None

    scores = region_info["scores"]
    X, Y, Z, C = heart_surface(scores, phase)

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=X,
            y=Z,
            z=Y,
            surfacecolor=C,
            colorscale="Reds",
            opacity=0.88,
            showscale=True,
            colorbar=dict(title="activation / evidence"),
            hoverinfo="skip",
        )
    )

    for region, coord in REGION_COORDS.items():
        x, y, z = coord
        score = scores.get(region, 0.0)
        leads = ", ".join(LEAD_REGIONS[region])

        fig.add_trace(
            go.Scatter3d(
                x=[x],
                y=[z],
                z=[y],
                mode="markers+text",
                text=[region],
                textposition="top center",
                marker=dict(
                    size=8 + 18 * score,
                    color=[score],
                    cmin=0,
                    cmax=1,
                    colorscale="Viridis",
                    opacity=0.95,
                ),
                name=region,
                hovertemplate=(
                    f"<b>{region}</b><br>"
                    f"score={score:.3f}<br>"
                    f"leads={leads}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=620,
        title=f"3D/4D CardioTwin heart map — {phase_label}",
        scene=dict(
            xaxis_title="left-right",
            yaxis_title="depth",
            zaxis_title="base-apex",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)
    return fig


def build_case_payload(result, phase_label, start_sec, end_sec):
    return {
        "version": "CardioTwin-AI v3.0.1 realtime replay heart demo",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "window_sec": [start_sec, end_sec],
        "phase_label": phase_label,
        "sqi": result["sqi"],
        "probabilities": result["probabilities"],
        "thresholds": result["thresholds"],
        "positive_labels": result["positive_labels"],
        "abnormal_positive_labels": result["abnormal_positive_labels"],
        "uncertain": result["uncertain"],
        "region_decision": {
            "decision": result["region"]["decision"],
            "reason": result["region"]["reason"],
            "top_region": result["region"]["top_region"],
            "top_score": result["region"]["top_score"],
            "second_region": result["region"]["second_region"],
            "second_score": result["region"]["second_score"],
            "margin": result["region"]["margin"],
            "scores": result["region"]["scores"],
        },
        "recommendation": result["recommendation"],
        "claim_boundary": "Research-use realtime replay visual demo. Not final diagnosis.",
    }


st.title("CardioTwin-AI v3.0.1 Real-time Replay + 3D/4D Heart Map")
st.caption(
    "Research-use demo. This version adds a live 3D/4D heart map to the replay dashboard. "
    "The current AI panel is pseudo/simulation until connected to the frozen v2.7 InceptionTime inference."
)

with st.sidebar:
    st.header("Replay Controls")
    uploaded = st.file_uploader("Optional: upload 12-lead ECG CSV", type=["csv"])
    fs = st.number_input("Sampling rate (Hz)", min_value=50, max_value=1000, value=500, step=50)
    demo_mode = st.selectbox("Synthetic demo pattern", ["mild", "sttc", "inferior", "lateral"])
    window_sec = st.slider("Replay window seconds", 2, 10, 6)
    step_sec = st.slider("Step seconds", 1, 5, 1)
    max_steps = st.slider("Max replay steps", 5, 90, 25)
    speed = st.slider("Replay delay seconds", 0.0, 2.0, 0.25, 0.05)
    start_button = st.button("Start replay + heart animation")

if uploaded:
    df = load_uploaded_csv(uploaded, fs=fs)
else:
    df = synthetic_ecg(fs=fs, seconds=24, abnormal_mode=demo_mode)

st.sidebar.write(f"Rows: {len(df)}")
st.sidebar.write(f"Duration: {df['time_sec'].max():.2f} sec")

plot_slot = st.empty()
heart_slot = st.empty()
panel_slot = st.empty()
table_slot = st.empty()
export_slot = st.empty()

def render_step(start_idx, end_idx):
    win = df.iloc[start_idx:end_idx]
    now_t = float(win["time_sec"].iloc[-1])
    phase_label, ph = phase_name(now_t)

    result = pseudo_ai_panel(win)
    payload = build_case_payload(
        result=result,
        phase_label=phase_label,
        start_sec=float(win["time_sec"].iloc[0]),
        end_sec=float(win["time_sec"].iloc[-1]),
    )

    left, right = st.columns([1.25, 1.0])

    with left:
        plot_ecg_window(df, start_idx, end_idx)

    with right:
        plot_heart_map(result["region"], ph, phase_label)

    with panel_slot.container():
        st.subheader("Live Safety / Review Panel")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("SQI", f"{result['sqi']:.3f}")
        c2.metric("Uncertain", str(result["uncertain"]))
        c3.metric("Positive", ", ".join(result["positive_labels"]) or "none")
        c4.metric("Abnormal", ", ".join(result["abnormal_positive_labels"]) or "none")
        c5.metric("Region", result["region"]["decision"])

        st.info(result["recommendation"])

    with table_slot.container():
        t1, t2 = st.tabs(["Prediction table", "Lead-region evidence"])

        with t1:
            pred_df = pd.DataFrame([
                {
                    "class": k,
                    "probability": v,
                    "threshold": result["thresholds"][k],
                    "positive": v >= result["thresholds"][k],
                }
                for k, v in result["probabilities"].items()
            ])
            st.dataframe(pred_df, use_container_width=True)

        with t2:
            region_df = pd.DataFrame([
                {
                    "region": region,
                    "linked_leads": ", ".join(LEAD_REGIONS[region]),
                    "evidence_score": score,
                }
                for region, score in result["region"]["scores"].items()
            ]).sort_values("evidence_score", ascending=False)
            st.dataframe(region_df, use_container_width=True)

            st.caption(
                f"Top={result['region']['top_region']} "
                f"Second={result['region']['second_region']} "
                f"Margin={result['region']['margin']:.3f} "
                f"Reason={result['region']['reason']}"
            )

    with export_slot.container():
        json_payload = json.dumps(payload, indent=2, ensure_ascii=False)
        html_report = f"""
        <!doctype html>
        <html>
        <head><meta charset="utf-8"><title>CardioTwin-AI v3.0.1 Replay Report</title></head>
        <body style="font-family:Arial;margin:32px;line-height:1.45">
        <h1>CardioTwin-AI v3.0.1 Replay + Heart Map Report</h1>
        <p><b>Boundary:</b> Research-use replay visual demo. Not final diagnosis.</p>
        <h2>Window</h2>
        <p>{payload["window_sec"][0]:.2f}–{payload["window_sec"][1]:.2f} sec</p>
        <h2>Phase</h2>
        <p>{payload["phase_label"]}</p>
        <h2>Safety</h2>
        <pre>{json_payload}</pre>
        </body>
        </html>
        """

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download v3.0.1 JSON replay report",
                data=json_payload.encode("utf-8"),
                file_name="cardiotwin_v301_replay_heart_report.json",
                mime="application/json",
            )
        with d2:
            st.download_button(
                "Download v3.0.1 HTML replay report",
                data=html_report.encode("utf-8"),
                file_name="cardiotwin_v301_replay_heart_report.html",
                mime="text/html",
            )


if not start_button:
    with plot_slot.container():
        render_step(0, min(len(df), int(window_sec * fs)))
else:
    n_window = int(window_sec * fs)
    n_step = int(step_sec * fs)

    for k, start in enumerate(range(0, max(1, len(df) - n_window), n_step)):
        if k >= max_steps:
            break
        end = min(len(df), start + n_window)

        plot_slot.empty()
        heart_slot.empty()
        panel_slot.empty()
        table_slot.empty()
        export_slot.empty()

        with plot_slot.container():
            render_step(start, end)

        time.sleep(speed)
'''

Path("apps").mkdir(exist_ok=True)
Path("apps/streamlit_realtime_replay_v301_heart.py").write_text(app, encoding="utf-8")

Path("artifacts/realtime_demo_v30").mkdir(parents=True, exist_ok=True)
Path("artifacts/realtime_demo_v30/REALTIME_REPLAY_V301_HEART_ADDENDUM.md").write_text(
    """# CardioTwin-AI v3.0.1 Real-time Replay + 3D/4D Heart Map Addendum

This addendum adds a new Streamlit dashboard:

`apps/streamlit_realtime_replay_v301_heart.py`

## Purpose

The original v3.0 replay dashboard provides ECG scrolling replay and a pseudo safety panel.  
v3.0.1 adds a live 3D/4D CardioTwin heart map, region evidence visualization, lead-to-region linking, P-QRS-ST-T phase animation, and replay report export.

## Claim Boundary

This dashboard is a research-use visual replay demo.  
The current AI/safety panel is pseudo/simulation until connected to the frozen v2.7 InceptionTime inference stack.

## Outputs

- Real-time 12-lead ECG replay
- 3D/4D heart map
- Region evidence table
- Prediction table
- JSON replay report
- HTML replay report
""",
    encoding="utf-8",
)

print("DONE: Created apps/streamlit_realtime_replay_v301_heart.py")
print("DONE: Created artifacts/realtime_demo_v30/REALTIME_REPLAY_V301_HEART_ADDENDUM.md")
