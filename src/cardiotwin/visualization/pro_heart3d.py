from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

REGION_SURFACE_CENTERS = {
    'septal': (0.00, -0.05, 0.10),
    'anterior': (0.00, -0.55, 0.05),
    'inferior': (0.00, 0.55, -0.32),
    'lateral': (0.62, 0.00, 0.00),
    'global_conduction': (-0.48, -0.03, 0.18),
    'global_rhythm': (0.00, 0.00, 0.78),
    'uncertain': (0.00, 0.00, -0.80),
}
REGION_TO_LEADS_TEXT = {
    'inferior': 'II, III, aVF',
    'septal': 'V1, V2',
    'anterior': 'V3, V4',
    'lateral': 'I, aVL, V5, V6',
    'global_conduction': 'QRS/global morphology',
    'global_rhythm': 'RR rhythm/global beat timing',
    'uncertain': 'Low confidence / low SQI / ambiguous evidence',
}

def _heart_like_surface(n=72):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    # heart-ish parametric shell: taper superior, widen lower LV visual zone
    rx = 0.62 * (1 + 0.18*np.cos(v))
    ry = 0.50 * (1 + 0.08*np.sin(2*v))
    rz = 0.96
    x = np.outer(np.cos(u), rx*np.sin(v))
    y = np.outer(np.sin(u), ry*np.sin(v))
    z = rz*np.outer(np.ones_like(u), np.cos(v))
    z = z - 0.12*np.abs(x) + 0.06*np.sin(np.outer(u, np.ones_like(v))*2)*np.sin(v)
    return x, y, z

def build_professional_heart3d(region_risk: dict, phase='ST', class_probs=None, safety=None):
    x,y,z = _heart_like_surface()
    # Heatmap scalar: smoothly emphasize nearby region centers on shell.
    heat = np.zeros_like(x, dtype=float)
    for region, center in REGION_SURFACE_CENTERS.items():
        risk = float(region_risk.get(region, 0.0))
        cx,cy,cz = center
        dist2 = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
        heat += risk * np.exp(-dist2 / 0.18)
    heat = heat / max(heat.max(), 1e-9)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, surfacecolor=heat, cmin=0, cmax=1,
        opacity=0.78, colorscale='Viridis', colorbar=dict(title='Region risk'),
        name='ECG-linked cardiac surface heatmap', hoverinfo='skip'
    ))
    for region, center in REGION_SURFACE_CENTERS.items():
        risk = float(region_risk.get(region, 0.0))
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]], mode='markers+text',
            marker=dict(size=8 + 26*risk, color=[risk], cmin=0, cmax=1, colorscale='Viridis', showscale=False),
            text=[f'{region}<br>risk={risk:.2f}<br>leads={REGION_TO_LEADS_TEXT.get(region,"-")}'],
            textposition='top center', name=region,
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    subtitle = f"phase={phase}"
    if safety:
        subtitle += f" · safety={safety.get('status','')} · confidence={safety.get('confidence_level','')}"
    fig.update_layout(
        title=f'Professional 3D/4D ECG-linked Cardiac Digital Twin · {subtitle}',
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
        height=720, margin=dict(l=0,r=0,t=50,b=0),
        legend=dict(orientation='h')
    )
    return fig

def build_phase_timeline_state(region_risk: dict, phase='ST'):
    phase = phase.upper()
    state = {'phase': phase, 'highlight_regions': [], 'description': ''}
    if phase == 'P':
        state['highlight_regions'] = ['global_rhythm']
        state['description'] = 'P phase proxy: rhythm/atrial-timing visual explanation.'
    elif phase == 'QRS':
        state['highlight_regions'] = ['global_conduction','septal']
        state['description'] = 'QRS phase proxy: ventricular depolarization and conduction emphasis.'
    elif phase == 'ST':
        state['highlight_regions'] = [r for r in ['inferior','anterior','septal','lateral'] if region_risk.get(r,0)>0.2]
        state['description'] = 'ST phase proxy: ischemia/injury-pattern region emphasis.'
    elif phase == 'T':
        state['highlight_regions'] = [r for r in ['inferior','anterior','septal','lateral'] if region_risk.get(r,0)>0.15]
        state['description'] = 'T phase proxy: repolarization/STTC region emphasis.'
    return state
