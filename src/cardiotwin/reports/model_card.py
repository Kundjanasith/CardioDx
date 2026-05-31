from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime


def write_model_card(out_path: str | Path, metrics: dict, manifest: dict | None = None, model_info: dict | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "model_name": "CardioTwin-AI 12L Baseline",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "intended_use": "Research-use preliminary 12-lead ECG screening and visual explanation.",
        "not_intended_for": "Final diagnosis, emergency triage, or replacement of clinician interpretation.",
        "inputs": "12-lead ECG array with leads I, II, III, aVR, aVL, aVF, V1-V6.",
        "outputs": "Multi-label probabilities for NORM, MI, STTC, CD, HYP and region-level visual explanation.",
        "metrics": metrics,
        "manifest_summary": manifest or {},
        "model_info": model_info or {},
        "limitations": [
            "Not patient-specific ECGI.",
            "Region-level mapping only.",
            "Dataset shift risk.",
            "Requires signal quality checks.",
            "Research-use only.",
        ],
    }
    out_path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
