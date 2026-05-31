from __future__ import annotations
import subprocess, sys
from pathlib import Path

def run(cmd):
    print('\n$ ' + ' '.join(cmd))
    subprocess.check_call([sys.executable] + cmd)

def main():
    if not Path('artifacts/processed/X_features.npy').exists():
        raise SystemExit('Missing artifacts/processed/X_features.npy. Run scripts/run_reproducible_baseline.py first.')
    if not Path('artifacts/models/baseline_model.joblib').exists():
        raise SystemExit('Missing artifacts/models/baseline_model.joblib. Train baseline first.')
    run(['scripts/build_safety_v12.py', '--processed-dir', 'artifacts/processed', '--model-path', 'artifacts/models/baseline_model.joblib'])
    run(['scripts/run_explainability_validation.py', '--processed-dir', 'artifacts/processed', '--model-path', 'artifacts/models/baseline_model_v12_safety.joblib', '--max-cases', '200'])
    run(['scripts/run_external_validation.py', '--ptbxl-processed-dir', 'artifacts/processed'])
    run(['scripts/build_research_evidence_pack.py'])
    print('\nDONE: Ultimate non-deep research pack completed.')
    print('Optional deep training: python scripts/train_deep_models.py --processed-dir artifacts/processed --epochs 5 --batch-size 128')
    print('Dashboard: streamlit run apps/safety_gate_dashboard.py')

if __name__ == '__main__': main()
