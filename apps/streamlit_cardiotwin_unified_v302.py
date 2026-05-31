
from __future__ import annotations

import json
import time
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


ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"

V27_DASHBOARD = ROOT / "apps" / "streamlit_dashboard_v27_export_pack.py"
V27_MODEL = ART / "models" / "inceptiontime_v21_safety.pt"
V27_THRESHOLDS = ART / "deep_safety_v21" / "threshold_profiles_deep.json"
V27_REGION_MAPPER = ROOT / "src" / "cardiotwin" / "explain" / "region_mapper_v23.py"
V27_RELEASE_MANIFEST = RELEASE / "release_manifest.json"

V28_MANIFEST = RELEASE / "cardiotwin_beatscope_v2_8_full_manifest.json"
V28_ADDENDUM = RELEASE / "BEATSCOPE_V28_RESEARCH_ADDENDUM.md"

V30_MANIFEST = ART / "v30_clinical_pilot_pack_manifest.json"
V30_PROTOCOL = ART / "prospective_pilot_v30" / "PROSPECTIVE_PILOT_PROTOCOL_v30.md"
V30_RISK = ART / "risk_management_v30" / "risk_register.csv"

OUT_DIR = ART / "unified_demo_v302"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    "septal": (0.00, 0.02, 0.08),
    "anterior": (0.02, -0.46, 0.00),
    "lateral": (0.58, -0.02, -0.10),
    "inferior": (0.02, 0.18, -0.78),
    "global_conduction": (-0.45, -0.02, 0.20),
    "hypertrophy_chamber": (0.35, 0.02, -0.38),
}


st.set_page_config(
    page_title="CardioTwin-AI v3.0.2 Unified Demo",
    layout="wide",
)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if limit:
        return txt[:limit]
    return txt


def artifact_status() -> dict:
    return {
        "v2.7 dashboard": str(V27_DASHBOARD.exists()),
        "v2.7 safety model": str(V27_MODEL.exists()),
        "v2.7 threshold profiles": str(V27_THRESHOLDS.exists()),
        "v2.7 region mapper v2.3": str(V27_REGION_MAPPER.exists()),
        "v2.7 release manifest": str(V27_RELEASE_MANIFEST.exists()),
        "v2.8 BeatScope manifest": str(V28_MANIFEST.exists()),
        "v2.8 research addendum": str(V28_ADDENDUM.exists()),
        "v3.0 pilot manifest": str(V30_MANIFEST.exists()),
        "v3.0 pilot protocol": str(V30_PROTOCOL.exists()),
        "v3.0 risk register": str(V30_RISK.exists()),
    }


def synthetic_ecg(fs=500, seconds=24, pattern="balanced"):
    t = np.arange(0, seconds, 1 / fs)
    data = {}

    for i, lead in enumerate(LEADS):
        p = np.zeros_like(t)
        qrs = np.zeros_like(t)
        tw = np.zeros_like(t)

        for beat in np.arange(0.65, seconds, 0.86):
            p += 0.06 * np.exp(-0.5 * ((t - (beat - 0.18)) / 0.035) ** 2)
            qrs += 0.90 * np.exp(-0.5 * ((t - beat) / 0.022) ** 2)
            qrs -= 0.24 * np.exp(-0.5 * ((t - (beat - 0.026)) / 0.014) ** 2)
            qrs -= 0.18 * np.exp(-0.5 * ((t - (beat + 0.035)) / 0.016) ** 2)
            tw += 0.20 * np.exp(-0.5 * ((t - (beat + 0.27)) / 0.085) ** 2)

        baseline = 0.04 * np.sin(2 * np.pi * 0.30 * t + i * 0.12)
        lead_gain = 1.0 - i * 0.018

        signal = lead_gain * (p + qrs + tw) + baseline

        if pattern == "inferior_mi_like" and lead in ["II", "III", "aVF"]:
            signal += 0.16 * np.exp(-0.5 * ((np.mod(t, 0.86) - 0.43) / 0.09) ** 2)
        elif pattern == "anterior_sttc_like" and lead in ["V3", "V4", "V5"]:
            signal += 0.10 * np.sin(2 * np.pi * 1.2 * t)
        elif pattern == "lateral_voltage_like" and lead in ["I", "aVL", "V5", "V6"]:
            signal *= 1.25
        elif pattern == "low_quality":
            signal += 0.15 * np.random.randn(len(t))

        signal += 0.006 * np.random.randn(len(t))
        data[lead] = signal

    df = pd.DataFrame(data)
    df.insert(0, "time_sec", t)
    return df


def load_uploaded_csv(file, fs):
    df = pd.read_csv(file)
    missing = [c for c in LEADS if c not in df.columns]
    if missing:
        st.warning(f"CSV missing leads {missing}. Falling back to synthetic ECG.")
        return synthetic_ecg(fs=fs)

    if "time_sec" not in df.columns:
        df.insert(0, "time_sec", np.arange(len(df)) / fs)

    return df[["time_sec"] + LEADS]


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


def estimate_sqi(df_window):
    x = df_window[LEADS].values.astype(float)
    finite = np.isfinite(x).mean()
    amp = float(np.nanmedian(np.abs(x)))
    noise = float(np.nanmedian(np.abs(np.diff(x, axis=0)))) if len(x) > 2 else 0.0
    flat = float((np.nanstd(x, axis=0) < 1e-4).mean())

    amp_score = np.clip(amp / 0.35, 0, 1)
    noise_score = 1 - np.clip(noise / 0.12, 0, 1)
    sqi = 0.50 * amp_score + 0.35 * noise_score + 0.15 * finite
    sqi *= 1 - 0.45 * flat
    return float(np.clip(sqi, 0, 1))


def lead_amplitudes(df_window):
    return {lead: float(np.nanmean(np.abs(df_window[lead].values.astype(float)))) for lead in LEADS}


def region_evidence(df_window):
    amp = lead_amplitudes(df_window)
    all_mean = np.mean(list(amp.values())) + 1e-9
    scores = {}

    for region, leads in LEAD_REGIONS.items():
        region_mean = np.mean([amp[l] for l in leads])
        rel = region_mean / all_mean
        scores[region] = float(np.clip((rel - 0.75) / 0.85, 0, 1))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    second, second_score = ranked[1]
    margin = top_score - second_score

    if top_score < 0.18:
        decision = "uncertain"
        reason = "low_region_evidence"
    elif margin < 0.08:
        decision = "uncertain"
        reason = "top_region_margin_too_small"
    else:
        decision = top
        reason = "dominant_region_evidence"

    return {
        "scores": scores,
        "lead_amplitudes": amp,
        "top_region": top,
        "top_score": top_score,
        "second_region": second,
        "second_score": second_score,
        "margin": margin,
        "decision": decision,
        "reason": reason,
    }


def _extract_threshold_number(value, default):
    """Extract numeric threshold from flexible JSON structures."""
    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return float(default)

    if isinstance(value, dict):
        # Common possible keys across threshold/calibration exports.
        for key in [
            "threshold",
            "value",
            "tuned_threshold",
            "selected_threshold",
            "operating_threshold",
            "cutoff",
            "cut_off",
            "decision_threshold",
        ]:
            if key in value:
                return _extract_threshold_number(value.get(key), default)

        # Sometimes profile stores a nested class entry.
        for key in ["screening", "balanced", "safety", "default"]:
            if key in value:
                return _extract_threshold_number(value.get(key), default)

        return float(default)

    return float(default)


def _threshold_dict_from_obj(obj, defaults):
    """Try to convert a candidate object into {class: threshold}."""
    if not isinstance(obj, dict):
        return None

    # Direct format:
    # {"NORM": 0.5, "MI": {"threshold": 0.3}, ...}
    if any(k in obj for k in defaults):
        return {
            k: _extract_threshold_number(obj.get(k, defaults[k]), defaults[k])
            for k in defaults
        }

    # Nested formats:
    # {"thresholds": {"NORM": ...}}
    # {"class_thresholds": {"NORM": ...}}
    # {"per_class": {"NORM": ...}}
    for key in [
        "thresholds",
        "class_thresholds",
        "per_class_thresholds",
        "per_class",
        "labels",
        "classes",
    ]:
        if key in obj and isinstance(obj[key], dict):
            found = _threshold_dict_from_obj(obj[key], defaults)
            if found is not None:
                return found

    return None


def load_thresholds():
    defaults = {
        "NORM": 0.50,
        "MI": 0.30,
        "STTC": 0.30,
        "CD": 0.25,
        "HYP": 0.30,
    }

    raw = read_json(V27_THRESHOLDS)

    if not raw:
        return defaults, "fallback_demo_thresholds"

    # 1) Try whole file directly.
    found = _threshold_dict_from_obj(raw, defaults)
    if found is not None:
        return found, "v2.7_thresholds:root"

    # 2) Try common top-level containers.
    for key in [
        "screening",
        "balanced",
        "safety",
        "default",
        "profiles",
        "threshold_profiles",
        "operating_profiles",
        "thresholds",
        "deep_thresholds",
    ]:
        if key not in raw:
            continue

        obj = raw[key]

        # If profiles contains screening profile, prefer it.
        if isinstance(obj, dict) and "screening" in obj:
            found = _threshold_dict_from_obj(obj["screening"], defaults)
            if found is not None:
                return found, f"v2.7_thresholds:{key}.screening"

        found = _threshold_dict_from_obj(obj, defaults)
        if found is not None:
            return found, f"v2.7_thresholds:{key}"

    # 3) Last resort: scan one level deeper.
    if isinstance(raw, dict):
        for key, obj in raw.items():
            if isinstance(obj, dict):
                found = _threshold_dict_from_obj(obj, defaults)
                if found is not None:
                    return found, f"v2.7_thresholds:auto_scan:{key}"

    return defaults, "threshold_file_found_but_unparsed_fallback"


def unified_ai_panel(df_window):
    sqi = estimate_sqi(df_window)
    region = region_evidence(df_window)
    thresholds, threshold_source = load_thresholds()

    s = region["scores"]
    probs = {
        "NORM": float(np.clip(0.74 * sqi - 0.18 * max(s.values()), 0.02, 0.98)),
        "MI": float(np.clip(0.08 + 0.48 * s["inferior"] + 0.30 * s["anterior"], 0.01, 0.96)),
        "STTC": float(np.clip(0.10 + 0.56 * s["anterior"], 0.01, 0.96)),
        "CD": float(np.clip(0.08 + 0.42 * s["global_conduction"] + 0.14 * (1 - sqi), 0.01, 0.96)),
        "HYP": float(np.clip(0.08 + 0.50 * s["hypertrophy_chamber"] + 0.08 * s["lateral"], 0.01, 0.96)),
    }

    positives = [k for k, v in probs.items() if v >= thresholds.get(k, 0.5)]
    abnormal = [k for k in positives if k != "NORM"]
    low_sqi = sqi < 0.55
    uncertain = low_sqi or region["decision"] == "uncertain"

    if low_sqi:
        recommendation = "Repeat ECG / doctor review due to low signal quality"
    elif "MI" in abnormal:
        recommendation = "Urgent doctor review for possible MI-like screening pattern"
    elif abnormal:
        recommendation = "Doctor review recommended"
    elif uncertain:
        recommendation = "Doctor review recommended due to uncertainty"
    else:
        recommendation = "Routine review"

    return {
        "mode": "unified_demo_with_v2.7_artifact_awareness",
        "sqi": sqi,
        "threshold_source": threshold_source,
        "probabilities": probs,
        "thresholds": thresholds,
        "positive_labels": positives,
        "abnormal_positive_labels": abnormal,
        "low_sqi": low_sqi,
        "uncertain": uncertain,
        "region": region,
        "recommendation": recommendation,
        "claim_boundary": "Demo inference panel. Use v2.7 dashboard/core for frozen model inference.",
    }


def plot_ecg(df, start, end):
    win = df.iloc[start:end]
    if not PLOTLY_READY:
        st.line_chart(win.set_index("time_sec")[LEADS])
        return

    fig = go.Figure()
    offset = 0.0
    for lead in LEADS:
        fig.add_trace(go.Scatter(
            x=win["time_sec"],
            y=win[lead] + offset,
            mode="lines",
            name=lead,
            line=dict(width=1),
        ))
        offset += 1.30

    fig.update_layout(
        height=540,
        title="Real-time 12-lead ECG replay",
        xaxis_title="Time (s)",
        yaxis_title="Lead offset",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def ellipsoid(cx, cy, cz, rx, ry, rz, nu=50, nv=28):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    U, V = np.meshgrid(u, v)
    X = cx + rx * np.cos(U) * np.sin(V)
    Y = cy + ry * np.sin(U) * np.sin(V)
    Z = cz + rz * np.cos(V)
    return X, Y, Z


def tube_z(cx, cy, z0, z1, r=0.09, nu=40, nz=18):
    u = np.linspace(0, 2 * np.pi, nu)
    z = np.linspace(z0, z1, nz)
    U, Z = np.meshgrid(u, z)
    X = cx + r * np.cos(U)
    Y = cy + r * np.sin(U)
    return X, Y, Z


def surface_intensity(X, Y, Z, region_scores, phase):
    # Activation wavefront from atria/base to apex.
    wave_center = 0.85 - 1.75 * phase
    wave = np.exp(-((Z - wave_center) ** 2) / 0.035)

    C = 0.15 + 0.55 * wave

    # Anatomical-style region weighting.
    anterior = np.exp(-((Y + 0.45) ** 2 + (Z - 0.00) ** 2 + X ** 2) / 0.22)
    inferior = np.exp(-((Y - 0.20) ** 2 + (Z + 0.72) ** 2 + X ** 2) / 0.20)
    lateral = np.exp(-((np.abs(X) - 0.48) ** 2 + (Z + 0.15) ** 2) / 0.20)
    septal = np.exp(-(X ** 2 + Y ** 2 + (Z - 0.03) ** 2) / 0.15)
    hyp = np.exp(-((X - 0.25) ** 2 + Y ** 2 + (Z + 0.36) ** 2) / 0.22)

    C += region_scores.get("anterior", 0) * anterior
    C += region_scores.get("inferior", 0) * inferior
    C += region_scores.get("lateral", 0) * lateral
    C += region_scores.get("septal", 0) * septal
    C += region_scores.get("hypertrophy_chamber", 0) * hyp
    C += 0.12 * region_scores.get("global_conduction", 0)

    return np.clip(C, 0, 1.5)


def plot_realistic_heart(result, phase, phase_label):
    if not PLOTLY_READY:
        st.info("Plotly not available.")
        return

    scores = result["region"]["scores"]
    fig = go.Figure()

    # Four-chamber anatomical-style approximation.
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
            name=name,
            hoverinfo="skip",
        ))

    # Great vessels: aorta / pulmonary trunk.
    vessels = [
        ("Aorta", 0.18, 0.00, 0.58, 1.12, 0.085, "Oranges"),
        ("Pulmonary artery", -0.18, 0.04, 0.56, 0.98, 0.075, "Blues"),
    ]

    for name, cx, cy, z0, z1, r, scale in vessels:
        X, Y, Z = tube_z(cx, cy, z0, z1, r=r)
        C = np.ones_like(X) * (0.35 + 0.45 * np.sin(np.pi * phase) ** 2)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=C,
            colorscale=scale,
            opacity=0.58,
            showscale=False,
            name=name,
            hoverinfo="skip",
        ))

    # Region markers.
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
            name=region,
            hovertemplate=f"<b>{region}</b><br>score={score:.3f}<br>leads={leads}<extra></extra>",
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


def beat_scope_summary():
    m = read_json(V28_MANIFEST)
    h = m.get("headline_results", {})
    return {
        "MIT-BIH best model": h.get("mitbih_best_model"),
        "MIT-BIH Macro-F1": h.get("mitbih_macro_f1"),
        "MIT-BIH AUROC": h.get("mitbih_auroc_macro"),
        "MIT-BIH AUPRC": h.get("mitbih_auprc_macro"),
        "PTBDB best model": h.get("ptbdb_best_model"),
        "PTBDB Macro-F1": h.get("ptbdb_macro_f1"),
        "PTBDB AUROC": h.get("ptbdb_auroc_macro"),
        "PTBDB AUPRC": h.get("ptbdb_auprc_macro"),
        "Transfer balanced accuracy gain": h.get("transfer_balanced_accuracy_gain"),
        "Transfer Macro-F1 gain": h.get("transfer_macro_f1_gain"),
        "Transfer AUROC gain": h.get("transfer_auroc_macro_gain"),
        "Transfer AUPRC gain": h.get("transfer_auprc_macro_gain"),
    }


def build_payload(result, phase_label, start_sec, end_sec):
    return {
        "version": "CardioTwin-AI v3.0.2 unified demo",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "window_sec": [start_sec, end_sec],
        "phase_label": phase_label,
        "v27_core_status": artifact_status(),
        "ai_safety": {
            "sqi": result["sqi"],
            "threshold_source": result["threshold_source"],
            "probabilities": result["probabilities"],
            "thresholds": result["thresholds"],
            "positive_labels": result["positive_labels"],
            "abnormal_positive_labels": result["abnormal_positive_labels"],
            "low_sqi": result["low_sqi"],
            "uncertain": result["uncertain"],
            "recommendation": result["recommendation"],
        },
        "region_mapper": result["region"],
        "beatscope_v28": beat_scope_summary(),
        "claim_boundary": "Unified research demo. Preliminary screening and visual explanation only. Not final diagnosis.",
    }


st.title("CardioTwin-AI v3.0.2 Unified Clinical Demo Dashboard")
st.caption(
    "Integrated world-class demo platform: v2.7 AI/safety/export core status + v2.8 BeatScope evidence + "
    "v3.0 clinical pilot workflow + real-time replay + anatomical-style 3D/4D heart twin."
)

with st.sidebar:
    st.header("Unified Demo Controls")
    fs = st.number_input("Sampling rate (Hz)", min_value=50, max_value=1000, value=500, step=50)
    pattern = st.selectbox(
        "Replay demo pattern",
        ["balanced", "inferior_mi_like", "anterior_sttc_like", "lateral_voltage_like", "low_quality"],
    )
    uploaded = st.file_uploader("Optional 12-lead CSV", type=["csv"])
    window_sec = st.slider("Replay window seconds", 2, 10, 6)
    step_sec = st.slider("Replay step seconds", 1, 5, 1)
    speed = st.slider("Replay delay seconds", 0.00, 1.50, 0.20, 0.05)
    max_steps = st.slider("Max replay steps", 5, 100, 25)

    st.divider()
    st.subheader("Core artifact status")
    st.json(artifact_status())

tabs = st.tabs([
    "Live Replay + Anatomical Heart Twin",
    "v2.7 AI/Safety Core",
    "BeatScope v2.8 Evidence",
    "Clinical Pilot / Doctor Review",
    "Unified Export Center",
])

if uploaded:
    df = load_uploaded_csv(uploaded, fs)
else:
    df = synthetic_ecg(fs=fs, seconds=24, pattern=pattern)

if "latest_payload" not in st.session_state:
    st.session_state.latest_payload = {}

with tabs[0]:
    st.subheader("Real-time ECG Replay + Anatomical-style 3D/4D CardioTwin")

    start_btn = st.button("Start unified replay", key="start_unified_replay")

    plot_container = st.empty()
    metrics_container = st.empty()
    tables_container = st.empty()

    def render(start, end):
        win = df.iloc[start:end]
        now_t = float(win["time_sec"].iloc[-1])
        phase_label, phase = phase_name(now_t)
        result = unified_ai_panel(win)

        c1, c2 = st.columns([1.18, 1.0])
        with c1:
            plot_ecg(df, start, end)
        with c2:
            plot_realistic_heart(result, phase, phase_label)

        with metrics_container.container():
            st.subheader("Unified Safety / Region / Review Panel")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("SQI", f"{result['sqi']:.3f}")
            m2.metric("Low SQI", str(result["low_sqi"]))
            m3.metric("Uncertain", str(result["uncertain"]))
            m4.metric("Abnormal", ", ".join(result["abnormal_positive_labels"]) or "none")
            m5.metric("Region", result["region"]["decision"])
            m6.metric("Phase", phase_label.split("/")[0].strip())
            st.info(result["recommendation"])
            st.caption(f"Threshold source: {result['threshold_source']}")

        with tables_container.container():
            tt1, tt2 = st.tabs(["Prediction", "Lead-region evidence"])
            with tt1:
                pred = pd.DataFrame([
                    {
                        "class": k,
                        "probability": v,
                        "threshold": result["thresholds"].get(k),
                        "positive": v >= result["thresholds"].get(k, 0.5),
                    }
                    for k, v in result["probabilities"].items()
                ])
                st.dataframe(pred, use_container_width=True)

            with tt2:
                region_df = pd.DataFrame([
                    {
                        "region": r,
                        "linked_leads": ", ".join(LEAD_REGIONS[r]),
                        "evidence_score": s,
                    }
                    for r, s in result["region"]["scores"].items()
                ]).sort_values("evidence_score", ascending=False)
                st.dataframe(region_df, use_container_width=True)
                st.caption(
                    f"Top={result['region']['top_region']} "
                    f"Second={result['region']['second_region']} "
                    f"Margin={result['region']['margin']:.3f} "
                    f"Reason={result['region']['reason']}"
                )

        payload = build_payload(
            result=result,
            phase_label=phase_label,
            start_sec=float(win["time_sec"].iloc[0]),
            end_sec=float(win["time_sec"].iloc[-1]),
        )
        st.session_state.latest_payload = payload

    if not start_btn:
        render(0, min(len(df), int(window_sec * fs)))
    else:
        n_window = int(window_sec * fs)
        n_step = int(step_sec * fs)
        for k, start in enumerate(range(0, max(1, len(df) - n_window), n_step)):
            if k >= max_steps:
                break
            end = min(len(df), start + n_window)
            plot_container.empty()
            metrics_container.empty()
            tables_container.empty()
            with plot_container.container():
                render(start, end)
            time.sleep(speed)

with tabs[1]:
    st.subheader("v2.7 AI / Safety / Region / Export Core")
    st.write("This tab keeps v2.7 as the frozen core and surfaces its artifact status inside the unified dashboard.")

    status_df = pd.DataFrame([
        {"artifact": k, "available": v}
        for k, v in artifact_status().items()
    ])
    st.dataframe(status_df, use_container_width=True)

    st.markdown("### Recommended command to open full v2.7 dashboard")
    st.code(r"& $PY -m streamlit run apps\streamlit_dashboard_v27_export_pack.py --server.port 8507", language="powershell")

    manifest = read_json(V27_RELEASE_MANIFEST)
    if manifest:
        st.markdown("### v2.7 release manifest summary")
        st.json({
            "release": manifest.get("release"),
            "title": manifest.get("title"),
            "files_indexed": manifest.get("files_indexed"),
            "high_level_metrics": manifest.get("high_level_metrics", {}),
        })
    else:
        st.warning("v2.7 release manifest not found or unreadable.")

    st.info(
        "For strict frozen-model inference, use the v2.7 dashboard/core. "
        "This unified dashboard is the integrated presentation layer."
    )

with tabs[2]:
    st.subheader("BeatScope v2.8 Evidence Tab")
    st.caption("Beat-level evidence is shown separately from 12-lead record-level validation.")

    bs = beat_scope_summary()
    st.dataframe(pd.DataFrame([{"metric": k, "value": v} for k, v in bs.items()]), use_container_width=True)

    if V28_ADDENDUM.exists():
        st.markdown("### Research Addendum Preview")
        st.text_area("BEATSCOPE_V28_RESEARCH_ADDENDUM.md", read_text(V28_ADDENDUM, limit=5000), height=420)
    else:
        st.warning("BeatScope v2.8 research addendum not found.")

with tabs[3]:
    st.subheader("Clinical Pilot / Doctor-in-the-loop Review")
    st.caption("Pilot workflow for controlled research use. Not clinical deployment.")

    c1, c2 = st.columns(2)

    with c1:
        case_id = st.text_input("Case ID", "PILOT-DEMO-0001")
        reviewer = st.text_input("Reviewer ID", "reviewer_01")
        reviewer_labels = st.multiselect("Reviewer labels", ["NORM", "MI", "STTC", "CD", "HYP", "OTHER"])
        reviewer_action = st.selectbox("Reviewer action", ["routine_review", "doctor_review", "repeat_ecg", "urgent_referral", "override_ai", "adjudication_needed"])
        comments = st.text_area("Reviewer comments", "")

    with c2:
        st.markdown("### Risk policy")
        if V30_RISK.exists():
            st.dataframe(pd.read_csv(V30_RISK), use_container_width=True)
        else:
            st.warning("Risk register not found.")

    review_payload = {
        "case_id": case_id,
        "reviewer": reviewer,
        "reviewer_labels": reviewer_labels,
        "reviewer_action": reviewer_action,
        "comments": comments,
        "latest_ai_payload": st.session_state.get("latest_payload", {}),
    }

    st.download_button(
        "Download doctor review JSON",
        data=json.dumps(review_payload, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"{case_id}_doctor_review.json",
        mime="application/json",
    )

with tabs[4]:
    st.subheader("Unified Export Center")

    payload = st.session_state.get("latest_payload", {})
    if not payload:
        st.warning("Run or render a replay step first.")
    else:
        report = {
            "unified_payload": payload,
            "beatscope_summary": beat_scope_summary(),
            "artifact_status": artifact_status(),
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "claim_boundary": "Research-use unified demo. Preliminary screening and visual explanation only. Not final diagnosis.",
        }

        json_text = json.dumps(report, indent=2, ensure_ascii=False)

        html = f"""
        <!doctype html>
        <html>
        <head>
        <meta charset="utf-8">
        <title>CardioTwin-AI v3.0.2 Unified Report</title>
        <style>
        body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
        .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
        pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
        </style>
        </head>
        <body>
        <h1>CardioTwin-AI v3.0.2 Unified Demo Report</h1>
        <div class="warning">Research-use unified demo. Not final diagnosis.</div>
        <h2>Summary</h2>
        <p>Integrated v2.7 core status, v2.8 BeatScope evidence, v3.0 clinical workflow, realtime ECG replay, and anatomical-style 3D/4D heart twin.</p>
        <h2>Payload</h2>
        <pre>{json_text}</pre>
        </body>
        </html>
        """

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download unified JSON report",
                data=json_text.encode("utf-8"),
                file_name="cardiotwin_v302_unified_report.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                "Download unified HTML report",
                data=html.encode("utf-8"),
                file_name="cardiotwin_v302_unified_report.html",
                mime="text/html",
            )

        st.json(report)
