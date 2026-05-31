from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from cardiotwin.constants import PTBXL_SUPERCLASSES
from cardiotwin.models.baseline_ml import load_model, predict_proba
from cardiotwin.eval.calibration import fit_isotonic_per_class, apply_isotonic, tune_threshold_profiles, reliability_table, calibration_summary
from cardiotwin.eval.safety_v12 import safety_metrics

def _html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="table")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed-dir', default='artifacts/processed')
    ap.add_argument('--model-path', default='artifacts/models/baseline_model.joblib')
    ap.add_argument('--out-model', default='artifacts/models/baseline_model_v12_safety.joblib')
    ap.add_argument('--out-dir', default='artifacts/safety_v12')
    args = ap.parse_args()
    processed = Path(args.processed_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X = np.load(processed/'X_features.npy')
    Y = np.load(processed/'Y_labels.npy')
    idx = pd.read_csv(processed/'records_index.csv')
    sqi = idx['sqi'].values if 'sqi' in idx.columns else np.ones(len(idx))
    train = idx['split'].values == 'train'
    val = idx['split'].values == 'val'
    test = idx['split'].values == 'test'
    bundle = load_model(args.model_path)
    model = bundle['model']; labels = bundle.get('label_names', PTBXL_SUPERCLASSES)
    raw_val = predict_proba(model, X[val])
    raw_test = predict_proba(model, X[test])
    calibrators, cal_val = fit_isotonic_per_class(Y[val], raw_val)
    cal_test = apply_isotonic(calibrators, raw_test)
    profiles, threshold_rows = tune_threshold_profiles(Y[val], cal_val, labels)
    bundle['calibration'] = {'method': 'isotonic_per_class_v12', 'calibrators': calibrators}
    bundle['threshold_profiles'] = profiles
    bundle['thresholds'] = profiles.get('balanced', {})
    bundle['v12_safety_gate'] = {
        'low_sqi_threshold': 0.55,
        'caution_sqi_threshold': 0.70,
        'uncertainty_margin': 0.08,
        'high_confidence_threshold': 0.85,
        'profiles': list(profiles.keys()),
        'clinical_boundary': 'Research-use preliminary screening and visual explanation only; not final diagnosis.',
    }
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out_model)

    rel = reliability_table(Y[test], cal_test, labels, n_bins=12)
    cal_sum = calibration_summary(Y[test], cal_test, labels)
    rel.to_csv(out/'calibration_reliability_bins.csv', index=False)
    cal_sum.to_csv(out/'calibration_summary.csv', index=False)
    threshold_rows.to_csv(out/'threshold_profiles_table.csv', index=False)
    (out/'threshold_profiles.json').write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding='utf-8')

    metrics_all = {}
    per_class_frames = []
    for profile in profiles:
        overall, per_class = safety_metrics(Y[test], cal_test, labels, profiles, sqi=sqi[test], profile=profile)
        metrics_all[profile] = overall
        per_class_frames.append(per_class)
    per_class_all = pd.concat(per_class_frames, ignore_index=True)
    per_class_all.to_csv(out/'safety_per_class_v12.csv', index=False)
    (out/'metrics_safety_v12.json').write_text(json.dumps(metrics_all, indent=2, ensure_ascii=False), encoding='utf-8')

    html = f"""
    <html><head><meta charset='utf-8'><title>CardioTwin-AI v1.2 Calibration and Safety Report</title>
    <style>body{{font-family:Arial, sans-serif;margin:36px;line-height:1.45}} .card{{border:1px solid #ddd;border-radius:10px;padding:18px;margin:16px 0}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid #eee;padding:7px;text-align:left}} .warn{{background:#fff3cd;padding:12px;border-radius:8px}}</style></head><body>
    <h1>CardioTwin-AI 12L v1.2 Calibration + Safety Gate Report</h1>
    <div class='warn'><b>Clinical boundary:</b> Research-use preliminary screening and visual explanation only; not final diagnosis.</div>
    <div class='card'><h2>Safety Metrics</h2><pre>{json.dumps(metrics_all, indent=2)}</pre></div>
    <div class='card'><h2>Threshold Profiles</h2>{_html_table(threshold_rows)}</div>
    <div class='card'><h2>Calibration Summary</h2>{_html_table(cal_sum)}</div>
    <div class='card'><h2>Reliability Bins</h2>{_html_table(rel)}</div>
    </body></html>
    """
    (out/'calibration_report.html').write_text(html, encoding='utf-8')
    print(json.dumps(metrics_all, indent=2))
    print(f'Saved calibrated safety model: {args.out_model}')
    print(f'Saved: {out/"metrics_safety_v12.json"}')
    print(f'Saved: {out/"threshold_profiles.json"}')
    print(f'Saved: {out/"calibration_report.html"}')

if __name__ == '__main__':
    main()
