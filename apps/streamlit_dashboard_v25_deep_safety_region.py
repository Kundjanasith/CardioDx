from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

try:
    import plotly.graph_objects as go
    PLOTLY_READY = True
except Exception:
    PLOTLY_READY = False

from cardiotwin.constants import PTBXL_SUPERCLASSES
from cardiotwin.signal.preprocessing import preprocess_ecg, pad_or_crop
from cardiotwin.models.deep_ecg import make_deep_model
from cardiotwin.explain.region_mapper_v23 import map_prediction_to_region


MODEL_PATH = Path("artifacts/models/inceptiontime_v21_safety.pt")
DEFAULT_RECORD = Path("data/raw/cinc2020/training/georgia/g1/E00001.hea")
FS_TARGET = 100.0
DURATION_SEC = 10.0
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@st.cache_resource
def load_geo_helpers():
    script_path = Path("scripts/evaluate_cinc2020_georgia_external.py")
    spec = importlib.util.spec_from_file_location("geo_eval", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource
def load_safety_model(path: str):
    ckpt = torch.load(path, map_location="cpu")
    model_name = ckpt.get("model_name", "inceptiontime")
    labels = ckpt.get("labels", PTBXL_SUPERCLASSES)

    model = make_deep_model(model_name, in_leads=12, n_classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    safety = ckpt.get("safety_v21", {})
    return model, model_name, labels, safety


def predict_tensor(model, x_lead_sample: np.ndarray) -> np.ndarray:
    xb = torch.tensor(x_lead_sample[None, :, :], dtype=torch.float32)
    with torch.no_grad():
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def load_record(hea_path: Path):
    geo = load_geo_helpers()
    mat_path = hea_path.with_suffix(".mat")
    if not hea_path.exists():
        raise FileNotFoundError(f"Missing .hea: {hea_path}")
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing .mat: {mat_path}")

    fs = geo.parse_fs(hea_path)
    sig = geo.load_signal(mat_path)
    x, fs2 = preprocess_ecg(sig, fs, target_fs=FS_TARGET, normalize=True)
    x = pad_or_crop(x, int(FS_TARGET * DURATION_SEC))
    dx_codes = geo.parse_dx(hea_path)
    return x.astype(np.float32), fs, dx_codes


def lead_occlusion_scores(model, x_time_lead: np.ndarray, class_index: int, full_probs: np.ndarray):
    x = x_time_lead.T.copy()  # [lead, sample]
    base_p = float(full_probs[class_index])
    scores = {}

    for i, lead in enumerate(LEADS):
        xo = x.copy()
        xo[i, :] = 0.0
        po = predict_tensor(model, xo)
        drop = max(0.0, base_p - float(po[class_index]))
        scores[lead] = drop

    total = sum(scores.values())
    if total > 0:
        scores = {k: float(v / total) for k, v in scores.items()}
    else:
        # fallback morphology proxy if occlusion is flat
        raw = np.mean(np.abs(x), axis=1)
        s = raw.sum()
        scores = {LEADS[i]: float(raw[i] / s) if s > 0 else 0.0 for i in range(len(LEADS))}
    return scores


def make_heart_3d(region_decisions):
    if not PLOTLY_READY:
        return None

    # Simple research/demo pseudo-heart mesh, not patient-specific anatomy.
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    x = 0.8 * np.outer(np.cos(u), np.sin(v))
    y = 1.0 * np.outer(np.sin(u), np.sin(v))
    z = 1.15 * np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        opacity=0.18,
        showscale=False,
        name="heart shell"
    ))

    region_xyz = {
        "anterior": (0.00, -0.85, 0.25),
        "septal": (0.00, -0.35, 0.20),
        "inferior": (0.00, 0.70, -0.55),
        "lateral": (0.72, 0.00, 0.05),
        "global_conduction": (0.00, 0.00, 0.00),
        "hypertrophy_chamber": (-0.65, 0.10, 0.10),
        "uncertain": (0.00, 0.00, 1.25),
    }

    rows = []
    for d in region_decisions:
        region = d.get("region", "uncertain")
        cls = d.get("predicted_class", "")
        prob = float(d.get("class_probability", 0.0))
        conf = float(d.get("confidence", 0.0))
        rx, ry, rz = region_xyz.get(region, region_xyz["uncertain"])
        rows.append((region, cls, prob, conf, rx, ry, rz, d.get("reason", "")))

    if rows:
        df = pd.DataFrame(rows, columns=["region", "class", "prob", "confidence", "x", "y", "z", "reason"])
        fig.add_trace(go.Scatter3d(
            x=df["x"], y=df["y"], z=df["z"],
            mode="markers+text",
            marker=dict(
                size=np.maximum(10, df["prob"] * 35),
                color=df["prob"],
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title="AI risk")
            ),
            text=df["class"] + "<br>" + df["region"],
            textposition="top center",
            customdata=df[["prob", "confidence", "reason"]],
            hovertemplate=(
                "Class=%{text}<br>"
                "Probability=%{customdata[0]:.3f}<br>"
                "Region confidence=%{customdata[1]:.3f}<br>"
                "Reason=%{customdata[2]}<extra></extra>"
            ),
            name="region risk"
        ))

    fig.update_layout(
        title="3D/4D CardioTwin Region Risk Map",
        scene=dict(
            xaxis_title="lateral",
            yaxis_title="anterior/inferior",
            zaxis_title="base/apex",
            aspectmode="data"
        ),
        height=640,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def run_inference(hea_path: Path, profile: str):
    model, model_name, labels, safety = load_safety_model(str(MODEL_PATH))
    x_time_lead, fs, dx_codes = load_record(hea_path)
    x_lead_sample = x_time_lead.T.copy()

    probs = predict_tensor(model, x_lead_sample)

    profiles = safety.get("threshold_profiles", {})
    selected = profiles.get(profile, {})

    rows = []
    region_decisions = []

    for i, label in enumerate(PTBXL_SUPERCLASSES):
        th = float(selected.get(label, {}).get("threshold", 0.5))
        pred = bool(probs[i] >= th)
        rows.append({
            "label": label,
            "probability": float(probs[i]),
            "threshold": th,
            "positive": pred,
        })

        if pred and label != "NORM":
            lead_scores = lead_occlusion_scores(model, x_time_lead, i, probs)
            decision = map_prediction_to_region(
                lead_scores=lead_scores,
                predicted_class=label,
                class_probability=float(probs[i]),
            )
            decision["lead_scores"] = lead_scores
            region_decisions.append(decision)

    return {
        "model_name": model_name,
        "record_id": hea_path.stem,
        "fs": fs,
        "dx_codes": dx_codes,
        "probs": probs,
        "prediction_table": pd.DataFrame(rows),
        "region_decisions": region_decisions,
        "signal": x_time_lead,
        "safety": safety,
    }


st.set_page_config(
    page_title="CardioTwin-AI v2.5 Deep Safety + 3D/4D Region Map",
    layout="wide"
)

st.title("CardioTwin-AI 12L v2.5")
st.subheader("Deep Safety + Region Mapper v2.3 + 3D/4D Heart Map")
st.caption("Research-use preliminary ECG screening and visual explanation prototype. Not for final diagnosis.")

if not MODEL_PATH.exists():
    st.error(f"Missing default safety model: {MODEL_PATH}")
    st.stop()

_, model_name, labels, safety = load_safety_model(str(MODEL_PATH))
profiles = list((safety.get("threshold_profiles") or {}).keys())
if not profiles:
    profiles = ["screening", "balanced", "high_specificity", "hyp_focus"]

colA, colB, colC = st.columns([2.2, 1, 1])
with colA:
    record_path = st.text_input("WFDB .hea path", str(DEFAULT_RECORD))
with colB:
    default_profile = safety.get("recommended_default_profile", "screening")
    idx = profiles.index(default_profile) if default_profile in profiles else 0
    profile = st.selectbox("Safety profile", profiles, index=idx)
with colC:
    phase = st.selectbox("4D phase view", ["P-wave", "QRS", "ST segment", "T-wave"], index=1)

if st.button("Run CardioTwin v2.5 inference", type="primary"):
    try:
        result = run_inference(Path(record_path), profile)

        st.success(f"Loaded model: {result['model_name']} | Record: {result['record_id']} | Profile: {profile}")

        m1, m2, m3, m4 = st.columns(4)
        pred_df = result["prediction_table"]
        positives = pred_df[pred_df["positive"]]["label"].tolist()

        m1.metric("Positive flags", len(positives))
        m2.metric("Max probability", f"{pred_df['probability'].max():.3f}")
        m3.metric("Region decisions", len(result["region_decisions"]))
        m4.metric("Sampling rate", f"{result['fs']} Hz")

        st.subheader("Safety-calibrated prediction")
        st.dataframe(pred_df, use_container_width=True)

        if positives:
            st.warning("Positive screening flags: " + ", ".join(positives))
        else:
            st.info("No class crossed the selected safety threshold.")

        st.subheader("12-lead ECG waveform")
        wave_df = pd.DataFrame(result["signal"], columns=LEADS)
        st.line_chart(wave_df)

        st.subheader("3D/4D CardioTwin Heart Region Map")
        fig = make_heart_3d(result["region_decisions"])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Plotly is not installed. Install with: pip install plotly")

        st.subheader("Region mapper v2.3 decisions")
        if result["region_decisions"]:
            compact = []
            for d in result["region_decisions"]:
                compact.append({
                    "class": d.get("predicted_class"),
                    "class_probability": d.get("class_probability"),
                    "region": d.get("region"),
                    "reason": d.get("reason"),
                    "confidence": d.get("confidence"),
                    "top_region": d.get("top_region", ""),
                    "second_region": d.get("second_region", ""),
                    "margin": d.get("margin", ""),
                })
            st.dataframe(pd.DataFrame(compact), use_container_width=True)

            with st.expander("Detailed lead evidence + region scores"):
                st.json(result["region_decisions"])
        else:
            st.info("No abnormal positive class selected for region mapping.")

        st.subheader("Safety metadata")
        st.json({
            "model_path": str(MODEL_PATH),
            "recommended_default_profile": safety.get("recommended_default_profile"),
            "calibration_note": safety.get("calibration_note"),
            "dx_codes": result["dx_codes"],
            "4d_phase_view": phase,
            "boundary": "Pseudo-3D/4D research visualization, not patient-specific ECGI anatomy."
        })

    except Exception as e:
        st.exception(e)

st.divider()
st.subheader("Available threshold profiles")
st.json(safety.get("threshold_profiles", {}))
