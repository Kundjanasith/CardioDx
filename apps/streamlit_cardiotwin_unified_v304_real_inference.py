
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_READY = True
except Exception:
    PLOTLY_READY = False

from cardiotwin.runtime.v304_real_inference_bridge import (
    LEADS,
    LEAD_REGIONS,
    PTBXL_LABELS,
    load_wfdb_hea_mat,
    load_csv_12lead,
    synthetic_ecg,
    run_v304_real_inference,
)

ROOT = Path(".")
ART = ROOT / "artifacts"
OUT_DIR = ART / "unified_demo_v304"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = ART / "models" / "inceptiontime_v21_safety.pt"
DEFAULT_THRESHOLDS = ART / "deep_safety_v21" / "threshold_profiles_deep.json"
DEFAULT_HEA = Path("data/raw/cinc2020/training/georgia/g1/E00001.hea")

REGION_COORDS = {
    "septal": (0.00, 0.02, 0.08),
    "anterior": (0.02, -0.46, 0.00),
    "lateral": (0.58, -0.02, -0.10),
    "inferior": (0.02, 0.18, -0.78),
    "global_conduction": (-0.45, -0.02, 0.20),
    "hypertrophy_chamber": (0.35, 0.02, -0.38),
}

st.set_page_config(page_title="CardioTwin-AI v3.0.4 Real Inference", layout="wide")


def make_time_df(x_12n, fs):
    t = np.arange(x_12n.shape[1]) / float(fs)
    df = pd.DataFrame(x_12n.T, columns=LEADS)
    df.insert(0, "time_sec", t)
    return df


def phase_name(t):
    cycle = 0.86
    ph = (t % cycle) / cycle
    if ph < 0.18:
        return "P wave / atrial activation", ph
    if ph < 0.32:
        return "QRS / ventricular depolarization", ph
    if ph < 0.55:
        return "ST segment", ph
    if ph < 0.78:
        return "T wave / ventricular repolarization", ph
    return "TP baseline", ph


def get_region_scores(result):
    rs = result.get("region_summary", {})
    if "scores" in rs:
        return rs["scores"]
    if "fallback_region" in rs and "scores" in rs["fallback_region"]:
        return rs["fallback_region"]["scores"]
    if "decisions" in rs and rs["decisions"]:
        d = rs["decisions"][0]
        if "region_scores" in d:
            return d["region_scores"]
    return {r: 0.0 for r in LEAD_REGIONS}


def get_region_decision(result):
    rs = result.get("region_summary", {})
    if "decision" in rs:
        return rs.get("decision", "uncertain")
    if "decisions" in rs and rs["decisions"]:
        first = rs["decisions"][0]
        return first.get("region", first.get("decision", "region_mapper_v23"))
    if "fallback_region" in rs:
        return rs["fallback_region"].get("decision", "uncertain")
    return "uncertain"


def plot_ecg(df):
    if not PLOTLY_READY:
        st.line_chart(df.set_index("time_sec")[LEADS])
        return

    fig = go.Figure()
    offset = 0.0
    for lead in LEADS:
        fig.add_trace(go.Scatter(
            x=df["time_sec"],
            y=df[lead] + offset,
            mode="lines",
            name=lead,
            line=dict(width=1),
        ))
        offset += 1.30

    fig.update_layout(
        height=560,
        title="Uploaded / Replay 12-lead ECG",
        xaxis_title="Time (s)",
        yaxis_title="Lead offset",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def ellipsoid(cx, cy, cz, rx, ry, rz, nu=48, nv=26):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    U, V = np.meshgrid(u, v)
    X = cx + rx * np.cos(U) * np.sin(V)
    Y = cy + ry * np.sin(U) * np.sin(V)
    Z = cz + rz * np.cos(V)
    return X, Y, Z


def tube_z(cx, cy, z0, z1, r=0.08, nu=36, nz=16):
    u = np.linspace(0, 2 * np.pi, nu)
    z = np.linspace(z0, z1, nz)
    U, Z = np.meshgrid(u, z)
    X = cx + r * np.cos(U)
    Y = cy + r * np.sin(U)
    return X, Y, Z


def surface_intensity(X, Y, Z, scores, phase):
    wave_center = 0.85 - 1.75 * phase
    wave = np.exp(-((Z - wave_center) ** 2) / 0.035)
    C = 0.15 + 0.55 * wave

    C += scores.get("anterior", 0) * np.exp(-((Y + 0.45) ** 2 + (Z - 0.00) ** 2 + X ** 2) / 0.22)
    C += scores.get("inferior", 0) * np.exp(-((Y - 0.20) ** 2 + (Z + 0.72) ** 2 + X ** 2) / 0.20)
    C += scores.get("lateral", 0) * np.exp(-((np.abs(X) - 0.48) ** 2 + (Z + 0.15) ** 2) / 0.20)
    C += scores.get("septal", 0) * np.exp(-(X ** 2 + Y ** 2 + (Z - 0.03) ** 2) / 0.15)
    C += scores.get("hypertrophy_chamber", 0) * np.exp(-((X - 0.25) ** 2 + Y ** 2 + (Z + 0.36) ** 2) / 0.22)
    C += 0.12 * scores.get("global_conduction", 0)

    return np.clip(C, 0, 1.5)


def plot_heart(result, phase, phase_label):
    if not PLOTLY_READY:
        st.info("Plotly not available.")
        return

    scores = get_region_scores(result)
    fig = go.Figure()

    parts = [
        ("Left ventricle", (0.22, 0.00, -0.38, 0.38, 0.36, 0.78, "Reds", 0.72)),
        ("Right ventricle", (-0.25, 0.04, -0.34, 0.32, 0.30, 0.66, "Blues", 0.52)),
        ("Left atrium", (0.24, 0.02, 0.46, 0.28, 0.26, 0.30, "Reds", 0.38)),
        ("Right atrium", (-0.25, 0.04, 0.44, 0.28, 0.25, 0.30, "Blues", 0.35)),
    ]

    for name, (cx, cy, cz, rx, ry, rz, scale, opacity) in parts:
        X, Y, Z = ellipsoid(cx, cy, cz, rx, ry, rz)
        C = surface_intensity(X, Y, Z, scores, phase)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=C,
            colorscale=scale,
            opacity=opacity,
            showscale=(name == "Left ventricle"),
            colorbar=dict(title="activation/evidence") if name == "Left ventricle" else None,
            hoverinfo="skip",
            name=name,
        ))

    for name, cx, cy, z0, z1, r, scale in [
        ("Aorta", 0.18, 0.00, 0.58, 1.12, 0.085, "Oranges"),
        ("Pulmonary artery", -0.18, 0.04, 0.56, 0.98, 0.075, "Blues"),
    ]:
        X, Y, Z = tube_z(cx, cy, z0, z1, r=r)
        C = np.ones_like(X) * (0.35 + 0.45 * np.sin(np.pi * phase) ** 2)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=C,
            colorscale=scale,
            opacity=0.58,
            showscale=False,
            hoverinfo="skip",
            name=name,
        ))

    for region, coord in REGION_COORDS.items():
        x, y, z = coord
        score = scores.get(region, 0)
        leads = ", ".join(LEAD_REGIONS[region])
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers+text",
            text=[region],
            textposition="top center",
            marker=dict(
                size=8 + 20 * score,
                color=[score],
                cmin=0,
                cmax=1,
                colorscale="Viridis",
                opacity=0.95,
            ),
            hovertemplate=f"<b>{region}</b><br>score={score:.3f}<br>leads={leads}<extra></extra>",
            name=region,
        ))

    fig.update_layout(
        height=620,
        title=f"Anatomical-style 3D/4D CardioTwin — {phase_label}",
        scene=dict(
            xaxis_title="left-right",
            yaxis_title="anterior-posterior",
            zaxis_title="base-apex",
            aspectmode="data",
            camera=dict(eye=dict(x=1.65, y=-1.8, z=1.25)),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def result_tables(result):
    pred_df = pd.DataFrame([
        {
            "class": k,
            "probability": result["probabilities"].get(k),
            "threshold": result["thresholds"].get(k),
            "positive": k in result["positive_labels"],
        }
        for k in PTBXL_LABELS
    ])

    scores = get_region_scores(result)
    region_df = pd.DataFrame([
        {
            "region": r,
            "linked_leads": ", ".join(LEAD_REGIONS[r]),
            "evidence_score": s,
        }
        for r, s in scores.items()
    ]).sort_values("evidence_score", ascending=False)

    return pred_df, region_df


st.title("CardioTwin-AI v3.0.4 Real Inference Bridge")
st.caption(
    "uploaded ECG / replay ECG → preprocess → inceptiontime_v21_safety.pt → threshold profiles → region_mapper_v23 → anatomical heart map → unified export"
)

with st.sidebar:
    st.header("Input")
    input_mode = st.selectbox("Input mode", ["WFDB path", "CSV upload", "Synthetic replay"])

    hea_path = st.text_input("WFDB .hea path", str(DEFAULT_HEA) if DEFAULT_HEA.exists() else "")
    uploaded_csv = st.file_uploader("12-lead CSV", type=["csv"])

    pattern = st.selectbox(
        "Synthetic pattern",
        ["balanced", "inferior_mi_like", "anterior_sttc_like", "lateral_voltage_like", "low_quality"],
    )

    fs = st.number_input("CSV/synthetic sampling rate", min_value=50, max_value=1000, value=500, step=50)

    st.divider()
    st.header("v2.7 Assets")
    model_path = st.text_input("Model path", str(DEFAULT_MODEL))
    threshold_path = st.text_input("Threshold profile path", str(DEFAULT_THRESHOLDS))
    profile = st.selectbox("Safety profile", ["screening", "balanced", "safety", "default"])
    device = st.selectbox("Device", ["cpu"])

    run_btn = st.button("Run v3.0.4 real inference bridge")

tabs = st.tabs(["Real Inference + Heart Map", "Bridge Diagnostics", "Region Mapper", "Unified Export"])

if "v304_result" not in st.session_state:
    st.session_state.v304_result = None
if "v304_df" not in st.session_state:
    st.session_state.v304_df = None

if run_btn:
    try:
        if input_mode == "WFDB path":
            x_raw, raw_fs, source_meta = load_wfdb_hea_mat(hea_path)
        elif input_mode == "CSV upload":
            if uploaded_csv is None:
                st.error("Please upload CSV first.")
                st.stop()
            x_raw, raw_fs, source_meta = load_csv_12lead(uploaded_csv, fs=fs)
        else:
            x_raw, raw_fs, source_meta = synthetic_ecg(fs=fs, seconds=10, pattern=pattern)

        result = run_v304_real_inference(
            x_raw=x_raw,
            fs=raw_fs,
            model_path=model_path,
            threshold_path=threshold_path,
            profile=profile,
            device=device,
            source_meta=source_meta,
        )

        st.session_state.v304_result = result
        st.session_state.v304_df = make_time_df(x_raw, raw_fs)

        out_path = OUT_DIR / "latest_v304_unified_result.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    except Exception as e:
        st.exception(e)

result = st.session_state.v304_result
df = st.session_state.v304_df

with tabs[0]:
    if result is None:
        st.info("Choose input and click 'Run v3.0.4 real inference bridge'.")
    else:
        phase_label, phase = phase_name(float(df["time_sec"].iloc[-1]))

        c1, c2 = st.columns([1.15, 1.0])
        with c1:
            plot_ecg(df)
        with c2:
            plot_heart(result, phase, phase_label)

        st.subheader("Safety-calibrated prediction")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Inference mode", result["inference_mode"])
        m2.metric("SQI", f"{result['sqi']:.3f}")
        m3.metric("Low SQI", str(result["low_sqi"]))
        m4.metric("Uncertain", str(result["uncertain"]))
        m5.metric("Abnormal", ", ".join(result["abnormal_positive_labels"]) or "none")
        m6.metric("Region", get_region_decision(result))

        st.info(result["recommendation"])

        pred_df, region_df = result_tables(result)
        t1, t2 = st.tabs(["Prediction table", "Region evidence"])
        with t1:
            st.dataframe(pred_df, use_container_width=True)
        with t2:
            st.dataframe(region_df, use_container_width=True)

with tabs[1]:
    if result is None:
        st.info("Run inference first.")
    else:
        st.subheader("Bridge diagnostics")
        st.json({
            "inference_mode": result.get("inference_mode"),
            "inference_error": result.get("inference_error"),
            "model_meta": result.get("model_meta"),
            "inference_meta": result.get("inference_meta"),
            "import_info": result.get("import_info"),
            "threshold_source": result.get("threshold_source"),
            "raw_shape": result.get("raw_shape"),
            "ai_shape": result.get("ai_shape"),
            "raw_fs": result.get("raw_fs"),
            "ai_fs": result.get("ai_fs"),
        })

with tabs[2]:
    if result is None:
        st.info("Run inference first.")
    else:
        st.subheader("Region mapper output")
        st.json(result.get("region_summary"))
        st.subheader("Region mapper meta")
        st.json(result.get("region_mapper_meta"))

with tabs[3]:
    if result is None:
        st.info("Run inference first.")
    else:
        export_payload = {
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "version": "CardioTwin-AI v3.0.4 real inference bridge export",
            "result": result,
            "claim_boundary": "Research-use preliminary screening support. Not final diagnosis.",
        }

        json_text = json.dumps(export_payload, indent=2, ensure_ascii=False)

        html = f"""
        <!doctype html>
        <html>
        <head>
        <meta charset="utf-8">
        <title>CardioTwin-AI v3.0.4 Real Inference Report</title>
        <style>
        body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
        .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
        pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
        </style>
        </head>
        <body>
        <h1>CardioTwin-AI v3.0.4 Real Inference Report</h1>
        <div class="warning">Research-use preliminary screening support. Not final diagnosis.</div>
        <h2>Summary</h2>
        <p><b>Inference mode:</b> {result.get("inference_mode")}</p>
        <p><b>Positive labels:</b> {", ".join(result.get("positive_labels", []))}</p>
        <p><b>Recommendation:</b> {result.get("recommendation")}</p>
        <h2>Full JSON Payload</h2>
        <pre>{json_text}</pre>
        </body>
        </html>
        """

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download v3.0.4 JSON report",
                data=json_text.encode("utf-8"),
                file_name="cardiotwin_v304_real_inference_report.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                "Download v3.0.4 HTML report",
                data=html.encode("utf-8"),
                file_name="cardiotwin_v304_real_inference_report.html",
                mime="text/html",
            )

        st.json(export_payload)
