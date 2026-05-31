from __future__ import annotations

import html
import json
import sys
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cardiotwin.constants import LEADS_12, PTBXL_SUPERCLASSES
from cardiotwin.explain.region_mapper_v23 import map_prediction_to_region
from cardiotwin.models.deep_ecg import make_deep_model
from cardiotwin.signal.preprocessing import pad_or_crop, preprocess_ecg

try:
    import plotly.graph_objects as go

    PLOTLY_READY = True
except Exception:
    PLOTLY_READY = False


MODEL_PATH = ROOT / "artifacts/models/inceptiontime_v21_safety.pt"
FS_TARGET = 100.0
DURATION_SEC = 10.0
WFDB_DEFAULT_FS = 500.0
PHASES = ["P-wave", "QRS", "ST segment", "T-wave"]
EXAMPLE_CSV_NAME = "synthetic_12lead_demo.csv"
EXAMPLE_NPZ_NAME = "synthetic_12lead_demo.npz"


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


def parse_wfdb_header(text: str) -> tuple[float, list[str], list[str]]:
    lines = text.splitlines()
    if not lines:
        raise ValueError("Empty WFDB header file.")

    first = lines[0].split()
    fs = float(first[2]) if len(first) >= 3 else WFDB_DEFAULT_FS
    leads = []
    dx_codes: list[str] = []

    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s[1:].strip()
            key, sep, value = body.partition(":")
            if sep and key.strip().lower() in {"dx", "diagnosis"}:
                tokens = [x.strip() for x in value.split(",") if x.strip()]
                dx_codes.extend(tokens)
            continue

        parts = s.split()
        if len(parts) >= 2:
            lead = parts[-1].strip()
            if lead.upper() == "AVR":
                lead = "aVR"
            elif lead.upper() == "AVL":
                lead = "aVL"
            elif lead.upper() == "AVF":
                lead = "aVF"
            leads.append(lead)

    if len(leads) != 12:
        leads = LEADS_12

    return fs, leads, dx_codes


def read_uploaded_csv(uploaded, fs_default: float):
    df = pd.read_csv(uploaded)
    cols = [c for c in LEADS_12 if c in df.columns]
    if len(cols) == 12:
        signal = df[cols].to_numpy(dtype=np.float32)
        leads = cols
    else:
        signal = df.select_dtypes(include="number").to_numpy(dtype=np.float32)
        leads = LEADS_12[: signal.shape[1]]

    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError("CSV must contain 12 numeric lead columns.")

    return signal, float(fs_default), leads, Path(uploaded.name).stem, []


def read_uploaded_npz(uploaded, fs_default: float):
    data = np.load(uploaded, allow_pickle=True)
    if "signal" not in data:
        raise ValueError("NPZ must contain a 'signal' array.")

    signal = np.asarray(data["signal"], dtype=np.float32)
    fs = float(data["fs"]) if "fs" in data else float(fs_default)
    leads = [str(x) for x in data["leads"]] if "leads" in data else LEADS_12[: signal.shape[1]]
    record_id = str(data["record_id"]) if "record_id" in data else Path(uploaded.name).stem

    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError("NPZ signal must have shape [samples, 12].")

    return signal, fs, leads, record_id, []


def read_uploaded_wfdb(hea_uploaded, mat_uploaded):
    header_text = hea_uploaded.getvalue().decode("utf-8-sig", errors="ignore")
    fs, leads, dx_codes = parse_wfdb_header(header_text)

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        hea_path = tmp / hea_uploaded.name
        mat_path = tmp / mat_uploaded.name
        hea_path.write_bytes(hea_uploaded.getvalue())
        mat_path.write_bytes(mat_uploaded.getvalue())

        data = loadmat(mat_path)
        if "val" not in data:
            raise ValueError("WFDB MAT file must contain a 'val' matrix.")

        signal = np.asarray(data["val"], dtype=np.float32)
        if signal.shape[0] == 12:
            signal = signal.T

    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError("WFDB MAT data must resolve to shape [samples, 12].")

    return signal, fs, leads, Path(hea_uploaded.name).stem, dx_codes


def load_record_from_inputs(input_mode: str, csv_file, npz_file, hea_file, mat_file, fs_default: float):
    if input_mode == "CSV upload":
        if csv_file is None:
            raise ValueError("Upload a 12-lead CSV file.")
        return read_uploaded_csv(csv_file, fs_default)

    if input_mode == "NPZ upload":
        if npz_file is None:
            raise ValueError("Upload a processed NPZ file.")
        return read_uploaded_npz(npz_file, fs_default)

    if hea_file is None or mat_file is None:
        raise ValueError("Upload both the WFDB .hea and .mat files.")
    return read_uploaded_wfdb(hea_file, mat_file)


def run_inference(signal: np.ndarray, fs: float, leads: list[str], record_id: str, dx_codes: list[str], profile: str):
    model, model_name, labels, safety = load_safety_model(str(MODEL_PATH))
    x_time_lead, _ = preprocess_ecg(signal, fs, target_fs=FS_TARGET, normalize=True)
    x_time_lead = pad_or_crop(x_time_lead, int(FS_TARGET * DURATION_SEC))
    x_lead_sample = x_time_lead.T.copy()

    probs = predict_tensor(model, x_lead_sample)

    profiles = safety.get("threshold_profiles", {})
    selected = profiles.get(profile, {})

    rows = []
    region_decisions = []

    for i, label in enumerate(labels):
        th = float(selected.get(label, {}).get("threshold", 0.5))
        pred = bool(probs[i] >= th)
        rows.append(
            {
                "label": label,
                "probability": float(probs[i]),
                "threshold": th,
                "positive": pred,
            }
        )

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
        "record_id": record_id,
        "raw_fs": fs,
        "dx_codes": dx_codes,
        "probs": probs,
        "prediction_table": pd.DataFrame(rows),
        "region_decisions": region_decisions,
        "signal": x_time_lead,
        "input_leads": leads,
        "safety": safety,
    }


def lead_occlusion_scores(model, x_time_lead: np.ndarray, class_index: int, full_probs: np.ndarray):
    x = x_time_lead.T.copy()
    base_p = float(full_probs[class_index])
    scores = {}

    for i, lead in enumerate(LEADS_12):
        xo = x.copy()
        xo[i, :] = 0.0
        po = predict_tensor(model, xo)
        drop = max(0.0, base_p - float(po[class_index]))
        scores[lead] = drop

    total = sum(scores.values())
    if total > 0:
        return {k: float(v / total) for k, v in scores.items()}

    raw = np.mean(np.abs(x), axis=1)
    raw_total = raw.sum()
    return {LEADS_12[i]: float(raw[i] / raw_total) if raw_total > 0 else 0.0 for i in range(len(LEADS_12))}


def make_heart_3d(region_decisions):
    if not PLOTLY_READY:
        return None

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    x = 0.8 * np.outer(np.cos(u), np.sin(v))
    y = 1.0 * np.outer(np.sin(u), np.sin(v))
    z = 1.15 * np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.18, showscale=False, name="heart shell"))

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
        fig.add_trace(
            go.Scatter3d(
                x=df["x"],
                y=df["y"],
                z=df["z"],
                mode="markers+text",
                marker=dict(
                    size=np.maximum(10, df["prob"] * 35),
                    color=df["prob"],
                    colorscale="Viridis",
                    opacity=0.9,
                    colorbar=dict(title="AI risk"),
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
                name="region risk",
            )
        )

    fig.update_layout(
        title="3D/4D CardioTwin Region Risk Map",
        scene=dict(
            xaxis_title="lateral",
            yaxis_title="anterior/inferior",
            zaxis_title="base/apex",
            aspectmode="data",
        ),
        height=640,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def fig_to_html_safe(fig):
    if fig is None:
        return "<p>3D figure unavailable.</p>"
    try:
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as exc:
        return f"<p>Could not export 3D figure: {html.escape(str(exc))}</p>"


def fig_to_png_bytes_safe(fig):
    if fig is None:
        return None
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def build_case_report_html(report_payload, prediction_table_html, region_table_html, fig_html):
    record_id = html.escape(str(report_payload.get("record_id", "unknown")))
    profile = html.escape(str(report_payload.get("safety_profile", "unknown")))
    model_name = html.escape(str(report_payload.get("model_name", "unknown")))
    boundary = html.escape(str(report_payload.get("boundary", "")))

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CardioTwin-AI Case Report - {record_id}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
h1, h2 {{ color: #1f2937; }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #eef2ff; margin-right: 8px; }}
.warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.small {{ color: #6b7280; font-size: 0.92em; }}
</style>
</head>
<body>
<h1>CardioTwin-AI 12L v2.7 Case Report</h1>
<p>
<span class="badge">Record: {record_id}</span>
<span class="badge">Model: {model_name}</span>
<span class="badge">Profile: {profile}</span>
</p>

<div class="warning">
<strong>Research-use boundary:</strong> {boundary}
</div>

<h2>Safety-calibrated Prediction</h2>
{prediction_table_html}

<h2>Region Mapper v2.3 Decisions</h2>
{region_table_html}

<h2>3D/4D CardioTwin Snapshot</h2>
<p class="small">Pseudo-3D/4D lead-region visual explanation only. Not patient-specific ECGI.</p>
{fig_html}

<h2>Machine-readable JSON Payload</h2>
<pre>{html.escape(json.dumps(report_payload, indent=2, ensure_ascii=False))}</pre>
</body>
</html>"""


def build_demo_signal(fs=500, duration=10, hr=72, noise=0.02, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs
    sig = np.zeros((len(t), 12), dtype=np.float32)
    period = 60 / hr
    r_times = np.arange(0.6, duration, period)
    for li in range(12):
        scale = 1.0 + 0.05 * li
        y = np.zeros_like(t)
        for r in r_times:
            y += 0.12 * scale * np.exp(-0.5 * ((t - (r - 0.18)) / 0.035) ** 2)
            y += -0.15 * scale * np.exp(-0.5 * ((t - (r - 0.025)) / 0.010) ** 2)
            y += 1.00 * scale * np.exp(-0.5 * ((t - r) / 0.012) ** 2)
            y += -0.25 * scale * np.exp(-0.5 * ((t - (r + 0.030)) / 0.012) ** 2)
            y += 0.35 * scale * np.exp(-0.5 * ((t - (r + 0.28)) / 0.065) ** 2)
        y += 0.05 * np.sin(2 * np.pi * 0.33 * t) + rng.normal(0, noise, len(t))
        if LEADS_12[li] in ["II", "III", "aVF"]:
            y += 0.03
        sig[:, li] = y
    return sig


def demo_csv_bytes() -> bytes:
    signal = build_demo_signal()
    df = pd.DataFrame(signal, columns=LEADS_12)
    return df.to_csv(index=False).encode("utf-8")


def demo_npz_bytes() -> bytes:
    signal = build_demo_signal()
    buf = BytesIO()
    np.savez_compressed(
        buf,
        signal=signal,
        fs=500.0,
        leads=np.array(LEADS_12),
        record_id="synthetic_demo",
    )
    return buf.getvalue()


st.set_page_config(page_title="CardioTwin-AI v2.7 Hosted Demo", layout="wide")
st.title("CardioTwin-AI 12L v2.7")
st.subheader("Deep Safety + Region Mapper v2.3 + 3D/4D Heart Map + Export Pack")
st.caption("Research-use preliminary ECG screening and visual explanation prototype. Not for final diagnosis.")

if not MODEL_PATH.exists():
    st.error(f"Missing bundled safety model: {MODEL_PATH}")
    st.stop()

_, model_name, labels, safety = load_safety_model(str(MODEL_PATH))
profiles = list((safety.get("threshold_profiles") or {}).keys())
if not profiles:
    profiles = ["screening", "balanced", "high_specificity", "hyp_focus"]

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("ECG source", ["CSV upload", "NPZ upload", "WFDB .hea + .mat"], index=0)
    fs_default = st.number_input("CSV default sampling rate (Hz)", min_value=50.0, max_value=1000.0, value=500.0, step=50.0)
    default_profile = safety.get("recommended_default_profile", "screening")
    profile_index = profiles.index(default_profile) if default_profile in profiles else 0
    profile = st.selectbox("Safety profile", profiles, index=profile_index)
    phase = st.selectbox("4D phase view", PHASES, index=1)
    st.divider()
    st.download_button("Download demo CSV", data=demo_csv_bytes(), file_name=EXAMPLE_CSV_NAME, mime="text/csv")
    st.download_button("Download demo NPZ", data=demo_npz_bytes(), file_name=EXAMPLE_NPZ_NAME, mime="application/octet-stream")

csv_file = None
npz_file = None
hea_file = None
mat_file = None

if input_mode == "CSV upload":
    csv_file = st.file_uploader("Upload 12-lead ECG CSV", type=["csv"])
elif input_mode == "NPZ upload":
    npz_file = st.file_uploader("Upload processed ECG NPZ", type=["npz"])
else:
    col1, col2 = st.columns(2)
    with col1:
        hea_file = st.file_uploader("Upload WFDB header (.hea)", type=["hea"])
    with col2:
        mat_file = st.file_uploader("Upload WFDB signal (.mat)", type=["mat"])

st.info(
    "Hosted deployment does not use local filesystem paths. Upload a CSV, NPZ, or WFDB .hea/.mat pair, "
    "or use the bundled demo downloads."
)

if st.button("Run CardioTwin v2.7 inference", type="primary"):
    try:
        signal, fs, leads, record_id, dx_codes = load_record_from_inputs(
            input_mode=input_mode,
            csv_file=csv_file,
            npz_file=npz_file,
            hea_file=hea_file,
            mat_file=mat_file,
            fs_default=fs_default,
        )
        result = run_inference(signal, fs, leads, record_id, dx_codes, profile)

        st.success(f"Loaded model: {result['model_name']} | Record: {result['record_id']} | Profile: {profile}")

        m1, m2, m3, m4 = st.columns(4)
        pred_df = result["prediction_table"]
        positives_all = pred_df[pred_df["positive"]]["label"].tolist()
        abnormal_flags = [x for x in positives_all if x != "NORM"]

        m1.metric("Abnormal flags", len(abnormal_flags))
        m2.metric("Max probability", f"{pred_df['probability'].max():.3f}")
        m3.metric("Region decisions", len(result["region_decisions"]))
        m4.metric("Sampling rate", f"raw {result['raw_fs']} Hz -> AI {FS_TARGET:.0f} Hz")

        st.subheader("Safety-calibrated prediction")
        st.dataframe(pred_df, use_container_width=True)

        if abnormal_flags:
            st.warning("Abnormal screening flags: " + ", ".join(abnormal_flags))
        elif "NORM" in positives_all:
            st.success("Only NORM crossed threshold. No abnormal class crossed the selected safety threshold.")
        else:
            st.info("No class crossed the selected safety threshold.")

        st.subheader("12-lead ECG waveform")
        wave_df = pd.DataFrame(result["signal"], columns=LEADS_12)
        wave_df.insert(0, "time_sec", np.arange(len(wave_df)) / FS_TARGET)
        st.line_chart(wave_df, x="time_sec", y=LEADS_12)

        st.subheader("3D/4D CardioTwin Heart Region Map")
        st.caption("Pseudo-3D/4D lead-region visual explanation only. This is not patient-specific ECGI or final diagnosis.")
        fig = make_heart_3d(result["region_decisions"])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Plotly is unavailable in this environment.")

        st.subheader("Region mapper v2.3 decisions")
        if result["region_decisions"]:
            compact = []
            for d in result["region_decisions"]:
                compact.append(
                    {
                        "class": d.get("predicted_class"),
                        "class_probability": d.get("class_probability"),
                        "region": d.get("region"),
                        "reason": d.get("reason"),
                        "confidence": d.get("confidence"),
                        "top_region": d.get("top_region", ""),
                        "second_region": d.get("second_region", ""),
                        "margin": d.get("margin", ""),
                    }
                )
            st.dataframe(pd.DataFrame(compact), use_container_width=True)

            with st.expander("Detailed lead evidence + region scores"):
                st.json(result["region_decisions"])
        else:
            st.info("No abnormal positive class selected for region mapping.")

        report_payload = {
            "model_path": str(MODEL_PATH.relative_to(ROOT)),
            "record_id": result["record_id"],
            "model_name": result["model_name"],
            "safety_profile": profile,
            "prediction_table": result["prediction_table"].to_dict(orient="records"),
            "region_decisions": result["region_decisions"],
            "recommended_default_profile": safety.get("recommended_default_profile"),
            "calibration_note": safety.get("calibration_note"),
            "dx_codes": result["dx_codes"],
            "input_leads": result["input_leads"],
            "4d_phase_view": phase,
            "boundary": "Pseudo-3D/4D research visualization, not patient-specific ECGI anatomy. Research-use only; not final diagnosis.",
        }

        st.subheader("Safety metadata")
        st.json(report_payload)

        st.download_button(
            "Download CardioTwin JSON report",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name=f"cardiotwin_v27_{result['record_id']}_{profile}.json",
            mime="application/json",
        )

        pred_html = result["prediction_table"].to_html(index=False)
        if result["region_decisions"]:
            region_export_df = pd.DataFrame(
                [
                    {
                        "class": d.get("predicted_class"),
                        "class_probability": d.get("class_probability"),
                        "region": d.get("region"),
                        "reason": d.get("reason"),
                        "confidence": d.get("confidence"),
                        "top_region": d.get("top_region", ""),
                        "second_region": d.get("second_region", ""),
                        "margin": d.get("margin", ""),
                    }
                    for d in result["region_decisions"]
                ]
            )
            region_html = region_export_df.to_html(index=False)
        else:
            region_html = "<p>No abnormal positive class selected for region mapping.</p>"

        fig_html = fig_to_html_safe(fig)
        html_report = build_case_report_html(
            report_payload=report_payload,
            prediction_table_html=pred_html,
            region_table_html=region_html,
            fig_html=fig_html,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Download HTML case report",
                data=html_report.encode("utf-8"),
                file_name=f"cardiotwin_v27_case_report_{result['record_id']}_{profile}.html",
                mime="text/html",
            )
        with c2:
            st.download_button(
                "Download interactive 3D HTML",
                data=fig_html.encode("utf-8"),
                file_name=f"cardiotwin_v27_3d_snapshot_{result['record_id']}_{profile}.html",
                mime="text/html",
            )
        with c3:
            png_bytes = fig_to_png_bytes_safe(fig)
            if png_bytes:
                st.download_button(
                    "Download 3D PNG snapshot",
                    data=png_bytes,
                    file_name=f"cardiotwin_v27_3d_snapshot_{result['record_id']}_{profile}.png",
                    mime="image/png",
                )
            else:
                st.caption("PNG snapshot requires kaleido. HTML snapshot is available.")

    except Exception as exc:
        st.exception(exc)

st.divider()
st.subheader("Available threshold profiles")
st.json(safety.get("threshold_profiles", {}))
