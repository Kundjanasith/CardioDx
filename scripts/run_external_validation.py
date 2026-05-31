from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def load_internal_metrics(processed_dir):
    idx = pd.read_csv(Path(processed_dir)/'records_index.csv')
    return {
        'dataset': 'PTB-XL_internal',
        'status': 'ready',
        'n_records': int(len(idx)),
        'n_train': int((idx['split']=='train').sum()),
        'n_val': int((idx['split']=='val').sum()),
        'n_test': int((idx['split']=='test').sum()),
        'label_columns_found': [c for c in ['NORM','MI','STTC','CD','HYP'] if c in idx.columns],
    }

def adapter_status(root, name, required_hint):
    p = Path(root) if root else None
    ready = bool(p and p.exists())
    return {
        'dataset': name,
        'status': 'adapter_ready_dataset_found' if ready else 'adapter_ready_dataset_not_found',
        'path': str(p) if p else '',
        'required_files_hint': required_hint,
        'note': 'Place dataset files here and rerun to perform true external validation.' if not ready else 'Dataset path detected; implement label harmonization before production claims.',
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ptbxl-processed-dir', default='artifacts/processed')
    ap.add_argument('--cinc2020-root', default='data/raw/cinc2020')
    ap.add_argument('--mimic-iv-ecg-root', default='data/raw/mimic_iv_ecg')
    ap.add_argument('--out-dir', default='artifacts/external_validation')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    datasets=[load_internal_metrics(args.ptbxl_processed_dir)]
    datasets.append(adapter_status(args.cinc2020_root, 'PhysioNet_CinC_2020', 'WFDB records + SNOMED/diagnosis metadata; harmonize to PTB-XL superclasses.'))
    datasets.append(adapter_status(args.mimic_iv_ecg_root, 'MIMIC_IV_ECG', 'Credentialed PhysioNet access; WFDB waveform paths + machine/clinical ECG reports; harmonize labels.'))
    shift_rows=[]
    internal_n=datasets[0]['n_records']
    for d in datasets:
        shift_rows.append({
            'dataset': d['dataset'],
            'status': d['status'],
            'n_records_observed': d.get('n_records', 0),
            'availability_gap_vs_ptbxl': max(0, internal_n - d.get('n_records', 0)) if d['dataset']!='PTB-XL_internal' else 0,
            'label_harmonization_required': d['dataset']!='PTB-XL_internal',
            'waveform_schema_required': d['dataset']!='PTB-XL_internal',
        })
    shift=pd.DataFrame(shift_rows)
    shift.to_csv(out/'dataset_shift_metrics.csv', index=False)
    gap={
        'ptbxl_internal_ready': True,
        'external_validation_included': any(d['status']=='adapter_ready_dataset_found' for d in datasets[1:]),
        'generalization_claim_allowed': any(d['status']=='adapter_ready_dataset_found' for d in datasets[1:]),
        'generalization_gap_summary': datasets,
        'honest_claim_boundary': 'Without downloaded CinC2020/MIMIC-IV-ECG, this run provides adapter readiness only, not true external validation performance.',
    }
    (out/'generalization_gap.json').write_text(json.dumps(gap, indent=2, ensure_ascii=False), encoding='utf-8')
    html=f"""
    <html><head><meta charset='utf-8'><title>External Validation Report</title>
    <style>body{{font-family:Arial;margin:36px}} table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid #eee;padding:8px;text-align:left}} .warn{{background:#fff3cd;padding:12px;border-radius:8px}}</style></head><body>
    <h1>CardioTwin-AI External Validation Pack</h1>
    <div class='warn'>{gap['honest_claim_boundary']}</div>
    {shift.to_html(index=False, escape=False)}
    <h2>Dataset Registry</h2><pre>{json.dumps(datasets, indent=2)}</pre>
    </body></html>
    """
    (out/'external_validation_report.html').write_text(html, encoding='utf-8')
    print(f'Saved {out/"external_validation_report.html"}')
    print(f'Saved {out/"dataset_shift_metrics.csv"}')
    print(f'Saved {out/"generalization_gap.json"}')

if __name__=='__main__':
    main()
