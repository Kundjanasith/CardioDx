from __future__ import annotations
import io, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from cardiotwin.constants import LEADS_12
from cardiotwin.models.baseline_ml import load_model
from cardiotwin.pipeline.inference import run_inference
from cardiotwin.reports.report_generator import generate_html_report, save_json_report
from cardiotwin.visualization.heart3d import build_heart3d_figure, phase_region_risk

st.set_page_config(page_title="CardioTwin-AI 12L", layout="wide")
st.title("CardioTwin-AI 12L")
st.caption("Low-cost 12-lead ECG AI screening + lead evidence + region-level 3D/4D visual explanation. Research-use only, not final diagnosis.")

MODEL_PATH = Path("artifacts/models/baseline_model.joblib")

@st.cache_resource
def get_model(path):
    return load_model(path)


def plot_ecg_plotly(signal, fs, leads):
    x = np.asarray(signal)
    n = min(len(x), int(fs*10))
    t = np.arange(n)/fs
    fig = go.Figure()
    offset = 0.0
    spacing = max(1.0, np.nanpercentile(np.abs(x[:n]), 95)*3)
    for i, lead in enumerate(leads):
        fig.add_trace(go.Scatter(x=t, y=x[:n, i] + offset, mode="lines", name=lead, line=dict(width=1)))
        offset -= spacing
    fig.update_layout(height=700, xaxis_title="Time (s)", yaxis_title="Leads offset", showlegend=True, margin=dict(l=20,r=20,t=20,b=20))
    return fig


def read_uploaded(uploaded, fs_default):
    name = uploaded.name.lower()
    if name.endswith(".npz"):
        data = np.load(uploaded, allow_pickle=True)
        signal = data["signal"].astype(np.float32)
        fs = float(data["fs"]) if "fs" in data else fs_default
        leads = [str(x) for x in data["leads"]] if "leads" in data else LEADS_12[:signal.shape[1]]
        rid = str(data["record_id"]) if "record_id" in data else uploaded.name
        return signal, fs, leads, rid
    df = pd.read_csv(uploaded)
    cols = [c for c in LEADS_12 if c in df.columns]
    if len(cols) == 12:
        signal = df[cols].values.astype(np.float32)
        leads = cols
    else:
        signal = df.select_dtypes(include="number").values.astype(np.float32)
        leads = LEADS_12[:signal.shape[1]]
    return signal, fs_default, leads, uploaded.name

with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model path", str(MODEL_PATH))
    fs_default = st.number_input("CSV sampling rate (Hz)", min_value=50.0, max_value=1000.0, value=500.0, step=50.0)
    st.warning("Train the baseline first if artifacts/models/baseline_model.joblib does not exist.")
    uploaded = st.file_uploader("Upload 12-lead ECG CSV or processed NPZ", type=["csv", "npz"])

if not Path(model_path).exists():
    st.error("Model not found. Run scripts/run_reproducible_baseline.py after placing PTB-XL under data/raw/ptbxl, or train with scripts/train_baseline.py.")
    st.stop()

bundle = get_model(model_path)

if uploaded is None:
    st.info("Upload a 12-lead ECG CSV/NPZ. You can create a demo file with: python scripts/make_synthetic_demo.py")
    st.stop()

signal, fs, leads, record_id = read_uploaded(uploaded, fs_default)
if signal.ndim != 2 or signal.shape[1] != 12:
    st.error(f"Expected 12 leads. Got shape {signal.shape}.")
    st.stop()

state = run_inference(bundle, signal, fs, leads, record_id=record_id)

c1, c2, c3 = st.columns([2.2, 1, 1])
with c1:
    st.subheader("12-lead ECG waveform")
    st.plotly_chart(plot_ecg_plotly(signal, fs, leads), use_container_width=True)
with c2:
    st.subheader("AI probabilities")
    probs = pd.DataFrame([{"class": k, "probability": v} for k, v in state["class_probabilities"].items()])
    st.dataframe(probs, use_container_width=True, hide_index=True)
    st.metric("Signal Quality", f"{state['sqi']['overall_sqi']:.3f}")
with c3:
    st.subheader("Region risk")
    regions = pd.DataFrame(state["regions"]["ranked_regions"])
    st.dataframe(regions, use_container_width=True, hide_index=True)
    st.metric("Top region", state["summary"]["top_region"], f"risk {state['summary']['top_region_risk']:.3f}")

st.subheader("Lead evidence")
lead_df = pd.DataFrame([{"lead": k, "importance": v} for k, v in sorted(state["lead_importance"].items(), key=lambda kv: kv[1], reverse=True)])
st.bar_chart(lead_df.set_index("lead"))

st.subheader("3D/4D Cardiac Digital Twin")
phase = st.select_slider("4D ECG phase", options=["P", "QRS", "ST", "T"], value="ST")
phase_risk = phase_region_risk(state["regions"]["region_risk"], phase)
st.plotly_chart(build_heart3d_figure(phase_risk, phase=phase), use_container_width=True)
with st.expander("Digital twin animation state JSON"):
    st.json(state["animation"])
st.caption("The 3D layer is a region-level visual explanation from 12-lead ECG evidence, not patient-specific ECGI.")

st.subheader("Export report")
out_dir = Path("artifacts/reports/dashboard")
out_dir.mkdir(parents=True, exist_ok=True)
json_path = save_json_report(state, out_dir / f"{record_id}_report.json")
html_path = generate_html_report(state, out_dir / f"{record_id}_report.html")
st.download_button("Download JSON report", data=json_path.read_bytes(), file_name=json_path.name, mime="application/json")
st.download_button("Download HTML report", data=html_path.read_bytes(), file_name=html_path.name, mime="text/html")

st.subheader("Clinical boundary")
st.warning(state["clinical_boundary"])
