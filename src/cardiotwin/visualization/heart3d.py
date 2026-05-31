from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

REGION_CENTERS = {
    "septal": (0.0, 0.0, 0.15),
    "anterior": (0.0, -0.55, 0.05),
    "inferior": (0.0, 0.55, -0.35),
    "lateral": (0.65, 0.0, 0.0),
    "global_conduction": (-0.45, 0.0, 0.2),
    "global_rhythm": (0.0, 0.0, 0.75),
    "uncertain": (0.0, 0.0, -0.85),
}

def _ellipsoid(rx=0.72, ry=0.55, rz=1.0, n=50):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z

def build_heart3d_figure(region_risk: dict, phase: str = "ST"):
    x, y, z = _ellipsoid()
    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.23, showscale=False, colorscale="Greys", name="generic heart"))
    for region, center in REGION_CENTERS.items():
        risk = float(region_risk.get(region, 0.0))
        size = 8 + 28*risk
        # do not set explicit colors; use numeric scale
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="markers+text",
            marker=dict(size=size, color=[risk], colorscale="Viridis", cmin=0, cmax=1, showscale=True if region == "septal" else False,
                        colorbar=dict(title="Risk") if region == "septal" else None),
            text=[f"{region}<br>{risk:.2f}"],
            textposition="top center",
            name=region,
        ))
    fig.update_layout(
        title=f"3D region-level cardiac visual explanation · phase={phase}",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

def phase_region_risk(base_region_risk: dict, phase: str):
    phase = phase.upper()
    out = dict(base_region_risk)
    if phase == "P":
        out["global_rhythm"] = max(out.get("global_rhythm", 0), 0.55)
    elif phase == "QRS":
        out["global_conduction"] = max(out.get("global_conduction", 0), 0.55)
    elif phase == "ST":
        pass
    elif phase == "T":
        # repolarization explanation emphasizes STTC-prone regions already present
        for k in ["anterior", "lateral", "inferior", "septal"]:
            out[k] = min(1.0, out.get(k, 0)*1.10)
    return out
