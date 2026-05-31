# CardioTwin-AI v3.4.7 PTB-XL Failure-case Review Addendum

Created: 2026-05-31T11:50:21.768457+00:00

## Purpose

This addendum reviews PTB-XL fold 10 label-level false positives and false negatives under runtime screening and fold-9-derived calibration policies.

## Inputs

- Prediction CSV: artifacts\public_multicenter_validation_v34\ptbxl_fold10_locked_inference_predictions_v344.csv
- Threshold picks: artifacts\public_multicenter_validation_v34\ptbxl_fold9_threshold_picks_v345.json
- Fold 10 OK rows: 2158

## Policy-level Failure Summary

### runtime_screening

- Total label events: 10790
- Failure events: 3537
- False positives: 3294
- False negatives: 243
- True positives: 2549
- True negatives: 4704

### fold9_best_f1

- Total label events: 10790
- Failure events: 1665
- False positives: 982
- False negatives: 683
- True positives: 2109
- True negatives: 7016

### fold9_sensitivity90

- Total label events: 10790
- Failure events: 2910
- False positives: 2605
- False negatives: 305
- True positives: 2487
- True negatives: 5393

## Calibration Impact

### Runtime screening → Fold-9 best-F1 threshold

- False-positive reduction: 2312
- False-negative change: 440
- Total failure-event reduction: 1872

### Runtime screening → Fold-9 sensitivity90 threshold

- False-positive reduction: 689
- False-negative change: 62
- Total failure-event reduction: 627

## Label-level Review

### runtime_screening

- NORM: FP=406, FN=51, Sensitivity=0.9470404984423676, Specificity=0.6602510460251046, Precision=0.6919575113808801
- MI: FP=327, FN=106, Sensitivity=0.8072727272727273, Specificity=0.7966417910447762, Precision=0.5758754863813229
- STTC: FP=454, FN=48, Sensitivity=0.9078694817658349, Specificity=0.7226634086744044, Precision=0.5102481121898598
- CD: FP=1095, FN=20, Sensitivity=0.9596774193548387, Specificity=0.34115523465703973, Precision=0.3029917250159134
- HYP: FP=1012, FN=18, Sensitivity=0.9312977099236641, Specificity=0.46624472573839665, Precision=0.1942675159235669

### fold9_best_f1

- NORM: FP=242, FN=147, Sensitivity=0.8473520249221184, Specificity=0.797489539748954, Precision=0.7712665406427222
- MI: FP=214, FN=160, Sensitivity=0.7090909090909091, Specificity=0.8669154228855721, Precision=0.6456953642384106
- STTC: FP=178, FN=126, Sensitivity=0.7581573896353166, Specificity=0.8912645082467929, Precision=0.6893542757417103
- CD: FP=196, FN=116, Sensitivity=0.7661290322580645, Specificity=0.8820697954271961, Precision=0.6597222222222222
- HYP: FP=152, FN=134, Sensitivity=0.48854961832061067, Specificity=0.919831223628692, Precision=0.45714285714285713

## Doctor-review Candidate Files

- Runtime top false positives: artifacts\public_multicenter_validation_v34\ptbxl_top_false_positive_cases_v347.csv
- Runtime top false negatives: artifacts\public_multicenter_validation_v34\ptbxl_top_false_negative_cases_v347.csv
- Best-F1 top false positives: artifacts\public_multicenter_validation_v34\ptbxl_top_false_positive_cases_bestf1_v347.csv
- Best-F1 top false negatives: artifacts\public_multicenter_validation_v34\ptbxl_top_false_negative_cases_bestf1_v347.csv

## Claim Boundary

Failure-case review is label-level retrospective analysis on PTB-XL fold 10. It identifies cases for doctor review but is not final diagnosis and not clinical deployment.