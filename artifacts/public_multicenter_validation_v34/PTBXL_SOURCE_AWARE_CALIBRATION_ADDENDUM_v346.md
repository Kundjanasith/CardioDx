# CardioTwin-AI v3.4.6 PTB-XL Source-aware Calibration Addendum

Created: 2026-05-31T11:38:09.001234+00:00

## Executive Summary

CardioTwin-AI v3.4 adds PTB-XL source-controlled benchmarking with fold 9 used for calibration and fold 10 used as locked evaluation. The frozen runtime showed strong fold-10 discrimination, and fold-9-derived thresholds improved specificity and F1 without retraining.

## Readiness

- Readiness: ready_candidate
- Metadata rows: 21799
- HR waveform-ready rows: 21799
- Label-mapped rows: 21388

## Cohorts

### fold9_calibration
- Rows: 2146
- Unique patients: 1917
- Label counts: {'NORM': 955, 'MI': 540, 'STTC': 528, 'CD': 495, 'HYP': 268}

### fold10_locked_eval
- Rows: 2158
- Unique patients: 1877
- Label counts: {'NORM': 963, 'MI': 550, 'STTC': 521, 'CD': 496, 'HYP': 262}

### fold10_balanced_smoke_subset
- Rows: 235
- Unique patients: 218
- Label counts: {'NORM': 66, 'MI': 67, 'STTC': 78, 'CD': 66, 'HYP': 60}

## Locked Inference

- Smoke: 235 / 235 OK, errors=0
- Fold 9 calibration: 2146 / 2146 OK, errors=0
- Fold 10 locked eval: 2158 / 2158 OK, errors=0

## Key Fold 10 Results

### Runtime screening threshold
- Macro AUROC: 0.8841
- Macro AUPRC: 0.7314
- Macro F1: 0.5814
- Macro precision: 0.4551
- Macro sensitivity: 0.9106
- Macro specificity: 0.5974

### Fold-9-derived best-F1 thresholds applied to fold 10
- Macro AUROC: 0.8841
- Macro AUPRC: 0.7314
- Macro F1: 0.6774
- Macro precision: 0.6446
- Macro sensitivity: 0.7139
- Macro specificity: 0.8715

## Interpretation

- PTB-XL fold 10 achieved strong discrimination under the frozen runtime, with macro AUROC 0.8841 and macro AUPRC 0.7314.
- Fold-9-derived best-F1 thresholds improved fold-10 macro F1, precision, specificity, and accuracy compared with runtime screening thresholds, while reducing sensitivity as expected.
- Runtime screening thresholds preserved high sensitivity but produced more false positives, consistent with screening-mode behavior.

## Claim Boundary

PTB-XL v3.4.6 is a source-controlled benchmark and calibration addendum. It is not prospective clinical validation, not final diagnosis, and not clinical deployment.