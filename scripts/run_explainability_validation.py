from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from cardiotwin.models.baseline_ml import load_model
from cardiotwin.pipeline.inference import run_inference
from cardiotwin.explain.temporal_occlusion import temporal_occlusion_importance
from cardiotwin.explain.clinical_alignment import lead_region_agreement, clinical_rule_agreement
from cardiotwin.mapping.validated_region_mapper import region_confusion_from_cases

REGION_PROXY_BY_LABEL = {
    'MI': 'inferior',
    'STTC': 'anterior',
    'CD': 'global_conduction',
    'HYP': 'lateral',
    'NORM': 'uncertain',
}

def infer_true_region_from_labels(row):
    for label in ['MI','STTC','CD','HYP','NORM']:
        if int(row.get(label,0)) == 1:
            return REGION_PROXY_BY_LABEL[label]
    return 'uncertain'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processed-dir', default='artifacts/processed')
    ap.add_argument('--model-path', default='artifacts/models/baseline_model.joblib')
    ap.add_argument('--out-dir', default='artifacts/explainability_v22')
    ap.add_argument('--max-cases', type=int, default=300)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    idx=pd.read_csv(Path(args.processed_dir)/'records_index.csv')
    idx=idx[idx['split']=='test'].head(args.max_cases).copy()
    bundle=load_model(args.model_path)
    rows=[]; true_regions=[]; pred_regions=[]
    examples=[]
    for _,row in idx.iterrows():
        d=np.load(row['npz_path'], allow_pickle=True)
        signal=d['signal'].astype(np.float32); fs=float(d['fs']); leads=[str(x) for x in d['leads']]
        state=run_inference(bundle, signal, fs, leads, record_id=str(row['record_id']))
        temporal=temporal_occlusion_importance(bundle, signal, fs, leads)
        agreement=lead_region_agreement(state['lead_importance'], state['regions']['region_risk'])
        clinical=clinical_rule_agreement(state['class_probabilities'], state['regions']['region_risk'])
        true_region=infer_true_region_from_labels(row)
        pred_region=state['summary']['top_region']
        true_regions.append(true_region); pred_regions.append(pred_region)
        rows.append({
            'record_id': row['record_id'],
            'true_region_proxy': true_region,
            'pred_top_region': pred_region,
            'top_region_risk': state['summary']['top_region_risk'],
            'lead_region_agreement': agreement,
            'clinical_rule_agreement': clinical['overall_clinical_rule_agreement'],
            'sqi': state['sqi']['overall_sqi'],
            'safety_status': state.get('safety_gate',{}).get('status',''),
        })
        if len(examples)<5:
            examples.append({'record_id': str(row['record_id']), 'temporal_occlusion': temporal, 'state_summary': state['summary']})
    df=pd.DataFrame(rows)
    df.to_csv(out/'lead_region_agreement.csv', index=False)
    conf=region_confusion_from_cases(true_regions, pred_regions)
    validation={
        'n_cases': int(len(df)),
        'lead_region_agreement_mean': float(df['lead_region_agreement'].mean()) if len(df) else None,
        'clinical_rule_agreement_mean': float(df['clinical_rule_agreement'].mean()) if len(df) else None,
        'region_confusion_matrix': conf,
        'examples': examples,
        'method_boundary': 'Region labels are proxy labels derived from superclasses; this validates consistency, not patient-specific anatomical ground truth.',
    }
    (out/'region_mapping_validation.json').write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding='utf-8')
    html=f"""
    <html><head><meta charset='utf-8'><title>Explainability v2.2 Report</title><style>body{{font-family:Arial;margin:36px}} table{{border-collapse:collapse;width:100%}} th,td{{padding:7px;border-bottom:1px solid #eee}}</style></head><body>
    <h1>CardioTwin-AI Explainability + Region Validation v2.2</h1>
    <p>{validation['method_boundary']}</p>
    <h2>Summary</h2><pre>{json.dumps({k:v for k,v in validation.items() if k!='examples'}, indent=2)}</pre>
    <h2>Lead-region agreement cases</h2>{df.to_html(index=False, escape=False)}
    <h2>Temporal occlusion examples</h2><pre>{json.dumps(examples, indent=2)}</pre>
    </body></html>
    """
    (out/'explainability_report.html').write_text(html, encoding='utf-8')
    print(f'Saved {out/"explainability_report.html"}')
    print(f'Saved {out/"lead_region_agreement.csv"}')
    print(f'Saved {out/"region_mapping_validation.json"}')

if __name__=='__main__':
    main()
