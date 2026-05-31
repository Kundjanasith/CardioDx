from __future__ import annotations
import json, hashlib, zipfile
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

TRIPOD_AI_ITEMS = [
    'Title identifies AI prediction model', 'Structured abstract', 'Rationale and objectives', 'Source of data', 'Eligibility criteria', 'Outcome definition', 'Predictors/features', 'Sample size', 'Missing data handling', 'Model development procedure', 'Model performance measures', 'Calibration assessment', 'Validation approach', 'Interpretability/explainability', 'Limitations', 'Clinical use boundary', 'Reproducibility artifacts'
]
STARD_AI_ITEMS = [
    'AI index test described', 'Reference standard or label source described', 'Dataset selection and flow', 'External validation/generalizability', 'Bias/fairness/applicability considerations', 'Threshold setting described', 'Uncertainty/abstain handling described', 'Safety failure modes described'
]
DECIDE_AI_ITEMS = [
    'Intended clinical workflow stage', 'Human-AI interaction described', 'Training and onboarding plan', 'Error handling and fallback', 'Early-stage evaluation setting', 'User feedback collection plan', 'Risk mitigation and monitoring'
]

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def build_research_cards(out_dir: str | Path, project_root: str | Path = '.') -> dict:
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    created=datetime.now(timezone.utc).isoformat()
    model_card={
        'project': 'CardioTwin-AI 12L', 'version': 'Ultimate Research Platform', 'created_at': created,
        'intended_use': 'Research-use 12-lead ECG preliminary screening and visual explanation.',
        'not_intended_use': 'Not final diagnosis, not emergency decision-making, not patient-specific ECGI.',
        'inputs': '12-lead ECG waveform or derived features',
        'outputs': 'Multi-label probabilities, thresholded labels, safety status, lead evidence, region-level visual explanation',
        'known_limitations': ['Single-dataset training unless external datasets are added', 'Region mapping is explanatory not anatomical ground truth', 'Requires physician review'],
    }
    dataset_card={
        'primary_dataset': 'PTB-XL', 'external_validation': 'Adapters for PhysioNet/CinC 2020 and MIMIC-IV-ECG are included; performance requires local dataset access.',
        'split_policy': 'Use PTB-XL strat_fold: 1-8 train, 9 validation, 10 test.',
        'leakage_controls': ['patient-level split expected from PTB-XL recommended folds', 'records_index manifest exported'],
    }
    limitation_card={
        'clinical_boundary': 'Preliminary screening and education/research explanation only.',
        'not_ecgi': 'The 3D/4D layer is region-level ECG explanation, not patient-specific electrocardiographic imaging.',
        'hardware_safety': 'Any real-time human measurement requires battery power, isolation, enclosure, lead-off checks, and safety review.',
    }
    risk_management={
        'hazards': ['False reassurance', 'False alarm', 'Low quality signal interpreted as disease', 'Overclaiming localization'],
        'mitigations': ['SQI rejection', 'abstain gate', 'calibration report', 'clinical boundary warning', 'human review requirement'],
        'residual_risk': 'Research prototype; not cleared as a medical device.',
    }
    checklists={
        'TRIPOD_AI_alignment': [{'item':x,'status':'addressed_or_planned'} for x in TRIPOD_AI_ITEMS],
        'STARD_AI_alignment': [{'item':x,'status':'addressed_or_planned'} for x in STARD_AI_ITEMS],
        'DECIDE_AI_alignment': [{'item':x,'status':'addressed_or_planned'} for x in DECIDE_AI_ITEMS],
    }
    objs={'model_card':model_card,'dataset_card':dataset_card,'limitation_card':limitation_card,'risk_management_summary':risk_management,'reporting_checklists':checklists}
    for name,obj in objs.items(): write_json(out/f'{name}.json', obj)
    return objs

def build_manifest(out_dir: str | Path, include_roots=None):
    out=Path(out_dir); include_roots=include_roots or ['artifacts/metrics','artifacts/safety_v12','artifacts/explainability_v22','artifacts/external_validation']
    rows=[]
    for r in include_roots:
        p=Path(r)
        if not p.exists(): continue
        for f in p.rglob('*'):
            if f.is_file(): rows.append({'path':str(f),'sha256':sha256_file(f),'bytes':f.stat().st_size})
    df=pd.DataFrame(rows)
    df.to_csv(out/'evidence_manifest.csv', index=False)
    return df

def zip_evidence(out_dir: str | Path, zip_path: str | Path):
    out=Path(out_dir); zip_path=Path(zip_path); zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file(): z.write(f, arcname=f.relative_to(out))
    return zip_path
