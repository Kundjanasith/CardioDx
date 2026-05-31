from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from cardiotwin.constants import LEADS_12
from cardiotwin.models.baseline_ml import load_model
from cardiotwin.pipeline.inference import run_inference
from cardiotwin.visualization.pro_heart3d import build_professional_heart3d, build_phase_timeline_state
from cardiotwin.reports.report_generator import save_json_report, generate_html_report

st.set_page_config(page_title='CardioTwin-AI Safety Gate Dashboard', layout='wide')
st.title('CardioTwin-AI 12L · Safety Gate Dashboard v1.2')
st.caption('Safety-calibrated preliminary ECG screening + SQI-aware interpretation + region-level 3D/4D visual explanation. Research-use only.')

@st.cache_resource
def get_model(path): return load_model(path)

def read_uploaded(uploaded, fs_default):
    name=uploaded.name.lower()
    if name.endswith('.npz'):
        data=np.load(uploaded, allow_pickle=True)
        signal=data['signal'].astype(np.float32); fs=float(data['fs']) if 'fs' in data else fs_default
        leads=[str(x) for x in data['leads']] if 'leads' in data else LEADS_12[:signal.shape[1]]
        rid=str(data['record_id']) if 'record_id' in data else uploaded.name
        return signal, fs, leads, rid
    df=pd.read_csv(uploaded)
    cols=[c for c in LEADS_12 if c in df.columns]
    signal=df[cols].values.astype(np.float32) if len(cols)==12 else df.select_dtypes(include='number').values.astype(np.float32)
    leads=cols if len(cols)==12 else LEADS_12[:signal.shape[1]]
    return signal, fs_default, leads, uploaded.name

def plot_ecg(signal, fs, leads):
    n=min(len(signal), int(fs*10)); t=np.arange(n)/fs
    fig=go.Figure(); spacing=max(1.0, np.nanpercentile(np.abs(signal[:n]),95)*3); offset=0
    for lead_i, lead in enumerate(leads):
        fig.add_trace(go.Scatter(x=t, y=signal[:n, lead_i]+offset, mode='lines', name=lead, line=dict(width=1)))
        offset -= spacing
    fig.update_layout(height=620, xaxis_title='Time (s)', yaxis_title='Leads offset', margin=dict(l=20,r=20,t=20,b=20))
    return fig

with st.sidebar:
    st.header('Model')
    default='artifacts/models/baseline_model_v12_safety.joblib'
    if not Path(default).exists(): default='artifacts/models/baseline_model_v11_calibrated.joblib' if Path('artifacts/models/baseline_model_v11_calibrated.joblib').exists() else 'artifacts/models/baseline_model.joblib'
    model_path=st.text_input('Model path', default)
    profile=st.selectbox('Threshold profile', ['screening','balanced','high_specificity'], index=1)
    fs_default=st.number_input('CSV sampling rate (Hz)', min_value=50.0, max_value=1000.0, value=500.0, step=50.0)
    uploaded=st.file_uploader('Upload 12-lead ECG CSV or processed NPZ', type=['csv','npz'])

if not Path(model_path).exists():
    st.error('Model not found. Run scripts/build_safety_v12.py or use baseline_model.joblib.')
    st.stop()
bundle=get_model(model_path)
if uploaded is None:
    st.info('Upload a 12-lead ECG CSV/NPZ. Demo: demo_data/synthetic_12lead_demo.csv')
    st.stop()
signal, fs, leads, rid = read_uploaded(uploaded, fs_default)
if signal.ndim != 2 or signal.shape[1] != 12:
    st.error(f'Expected 12 leads. Got {signal.shape}.')
    st.stop()
state=run_inference(bundle, signal, fs, leads, rid, threshold_profile=profile)
safety=state.get('safety_gate', {})

status=safety.get('status','UNKNOWN')
if status.startswith('REJECT'):
    st.error(f"Safety gate: {status} — {safety.get('reason','')}")
elif status.startswith('ABSTAIN'):
    st.warning(f"Safety gate: {status} — {safety.get('reason','')}")
else:
    st.success(f"Safety gate: {status} · confidence={safety.get('confidence_level','')} · {safety.get('reason','')}")

c1,c2,c3=st.columns([2.0,1.0,1.0])
with c1:
    st.subheader('12-lead ECG waveform')
    st.plotly_chart(plot_ecg(signal, fs, leads), use_container_width=True)
with c2:
    st.subheader('Calibrated AI probabilities')
    st.dataframe(pd.DataFrame([{'class':k,'probability':v,'prediction':state['thresholded_prediction'].get(k,0)} for k,v in state['class_probabilities'].items()]), use_container_width=True, hide_index=True)
    st.metric('Signal Quality', f"{state['sqi']['overall_sqi']:.3f}")
    st.metric('Safety status', safety.get('status',''))
with c3:
    st.subheader('Region risk')
    st.dataframe(pd.DataFrame(state['regions']['ranked_regions']), use_container_width=True, hide_index=True)
    st.metric('Top region', state['summary']['top_region'], f"risk {state['summary']['top_region_risk']:.3f}")
    st.write('Predicted labels:', ', '.join(state.get('predicted_labels', [])) or 'None')

st.subheader('Lead evidence')
st.bar_chart(pd.DataFrame([{'lead':k,'importance':v} for k,v in sorted(state['lead_importance'].items(), key=lambda kv:kv[1], reverse=True)]).set_index('lead'))
st.subheader('Professional 3D/4D Cardiac Digital Twin')
phase=st.select_slider('4D ECG phase', options=['P','QRS','ST','T'], value='ST')
st.json(build_phase_timeline_state(state['regions']['region_risk'], phase), expanded=False)
st.plotly_chart(build_professional_heart3d(state['regions']['region_risk'], phase=phase, class_probs=state['class_probabilities'], safety=safety), use_container_width=True)

st.subheader('Export report')
out_dir=Path('artifacts/reports/safety_dashboard'); out_dir.mkdir(parents=True, exist_ok=True)
json_path=save_json_report(state, out_dir/f'{rid}_safety_report.json')
html_path=generate_html_report(state, out_dir/f'{rid}_safety_report.html')
st.download_button('Download JSON report', data=json_path.read_bytes(), file_name=json_path.name, mime='application/json')
st.download_button('Download HTML report', data=html_path.read_bytes(), file_name=html_path.name, mime='text/html')
st.warning('Clinical boundary: research-use preliminary screening and visual explanation only; not final diagnosis or patient-specific ECGI.')
