# CardioTwin-AI v3.4.5 PTB-XL Metrics + Source-aware Calibration

Created: 2026-05-31T11:36:20.063050+00:00

## Inputs

- Fold 9 calibration predictions: artifacts\public_multicenter_validation_v34\ptbxl_fold9_calibration_inference_predictions_v345.csv
- Fold 10 locked evaluation predictions: artifacts\public_multicenter_validation_v34\ptbxl_fold10_locked_inference_predictions_v344.csv
- Fold 9 OK rows: 2146
- Fold 10 OK rows: 2158

## Macro Summary

### fold10_apply_fold9_best_f1

- macro_auroc: 0.8840655307419931
- macro_auprc: 0.7313501038399431
- macro_f1: 0.6773663026278848
- macro_precision: 0.6446362519975846
- macro_sensitivity: 0.7138557948454038
- macro_specificity: 0.8715140979874414
- macro_accuracy: 0.845690454124189
- labels_with_valid_auroc: 5

### fold10_apply_fold9_best_youden

- macro_auroc: 0.8840655307419931
- macro_auprc: 0.7313501038399431
- macro_f1: 0.6565193768089721
- macro_precision: 0.5797313808155391
- macro_sensitivity: 0.8002203958930666
- macro_specificity: 0.8071044902883022
- macro_accuracy: 0.8040778498609823
- labels_with_valid_auroc: 5

### fold10_apply_fold9_sensitivity90_max_specificity

- macro_auroc: 0.8840655307419931
- macro_auprc: 0.7313501038399431
- macro_f1: 0.6139544436862824
- macro_precision: 0.4913253202746171
- macro_sensitivity: 0.8926916394625979
- macro_specificity: 0.6829833327639918
- macro_accuracy: 0.7303058387395737
- labels_with_valid_auroc: 5

### fold10_apply_fold9_sensitivity95_max_specificity

- macro_auroc: 0.8840655307419931
- macro_auprc: 0.7313501038399431
- macro_f1: 0.5561850147060081
- macro_precision: 0.41471653264446956
- macro_sensitivity: 0.9516459398008996
- macro_specificity: 0.5258403780798032
- macro_accuracy: 0.6251158480074144
- labels_with_valid_auroc: 5

### fold10_runtime_screening

- macro_auroc: 0.8840655307419931
- macro_auprc: 0.7313501038399431
- macro_f1: 0.5814457423805688
- macro_precision: 0.45506807017830864
- macro_sensitivity: 0.9106315673518864
- macro_specificity: 0.5973912412279443
- macro_accuracy: 0.6721964782205746
- labels_with_valid_auroc: 5

### fold9_runtime_screening

- macro_auroc: 0.8857478656133132
- macro_auprc: 0.7271438204468488
- macro_f1: 0.5860522621911997
- macro_precision: 0.4597555270714354
- macro_sensitivity: 0.9110521964454528
- macro_specificity: 0.6005317554664353
- macro_accuracy: 0.6751164958061511
- labels_with_valid_auroc: 5

## Claim Boundary

PTB-XL v3.4.5 is a source-controlled benchmark and calibration analysis. Not prospective clinical validation and not final diagnosis.