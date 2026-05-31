# CardioTwin-AI v3.3.5 Final Public Multicenter Validation Report

Created: 2026-05-24T21:19:50.575509+00:00

## Executive Summary

CardioTwin-AI was evaluated with a frozen runtime on a locked, source-separated public multi-center ECG cohort. The system showed promising discrimination across multiple public sources, with screening-oriented sensitivity and a need for source-aware threshold calibration and doctor-in-the-loop review before clinical claims.

## Locked Cohort

- Total records selected: 3222
- Sources:
  - cpsc_2018: 894 records
  - cpsc_2018_extra: 950 records
  - ptb: 415 records
  - georgia: 963 records

## Label Totals

- NORM: 682
- MI: 777
- STTC: 1351
- CD: 969
- HYP: 542

## Frozen Inference

- Records requested: 3222
- OK count: 3222
- Error count: 0
- Runtime seconds total: 319.84
- Device: cpu
- Profile: screening

## Source-separated Metrics

### cpsc_2018

- Macro AUROC: 0.8394
- Macro AUPRC: 0.7289
- Macro F1: 0.3760
- Macro sensitivity: 0.4877
- Macro specificity: 0.5956
- Macro precision: 0.3189
- Claim use: source_separated_result

### cpsc_2018_extra

- Macro AUROC: 0.8076
- Macro AUPRC: 0.5908
- Macro F1: 0.5030
- Macro sensitivity: 0.8911
- Macro specificity: 0.4968
- Macro precision: 0.4279
- Claim use: source_separated_result

### georgia

- Macro AUROC: 0.7546
- Macro AUPRC: 0.5306
- Macro F1: 0.5033
- Macro sensitivity: 0.7934
- Macro specificity: 0.5406
- Macro precision: 0.4061
- Claim use: source_separated_result

### ptb

- Macro AUROC: 0.9164
- Macro AUPRC: 0.7194
- Macro F1: 0.3615
- Macro sensitivity: 0.7324
- Macro specificity: 0.6249
- Macro precision: 0.3227
- Claim use: source_separated_result

### ALL_SOURCES_STACKED_REFERENCE_ONLY

- Macro AUROC: 0.8090
- Macro AUPRC: 0.6309
- Macro F1: 0.5465
- Macro sensitivity: 0.8523
- Macro specificity: 0.5581
- Macro precision: 0.4231
- Claim use: descriptive_only_not_random_split

## Calibration and Threshold Stress

v3.3.3 shows that screening thresholds favor sensitivity but increase false positives. Threshold recommendations are analytical only and do not modify the frozen runtime.

## Failure-case Review

- Total failure events: 6020
- False positives: 5302
- False negatives: 718
- High-confidence false positives: 1097
- Low-score false negatives: 616
- Low-SQI failure events: 2

## Recommended Next Actions

- Review top false positives and false negatives by source and label before any clinical claim.
- Prioritize doctor review for high-confidence false positives and low-score false negatives.
- Treat rare-label source results cautiously, especially source-label pairs with fewer than 30 positives.
- Use v3.3.3 threshold stress as analytical guidance only; do not overwrite frozen runtime thresholds yet.
- Add prospective doctor-in-the-loop adjudication before clinical deployment claims.
- Build v3.4 source-aware calibration pack.
- Run doctor-in-the-loop adjudication using the v3.3.4 template.
- Keep MIMIC-IV-ECG as an access-gated future validation path.

## Safe Claim

CardioTwin-AI was evaluated with a frozen runtime on a locked, source-separated public multi-center ECG cohort. The system showed promising discrimination across multiple public sources, with screening-oriented sensitivity and a need for source-aware threshold calibration and doctor-in-the-loop review before clinical claims.

## Disallowed Claims

- clinically validated
- doctor-level diagnosis
- ready for autonomous clinical diagnosis
- generalizes to every hospital
- MIMIC-IV-ECG validated
- prospectively validated

## Claim Boundary

Research-use public multi-center validation evidence only. Not prospective clinical validation, not final diagnosis, and not clinical deployment.