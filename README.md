# CardioTwin-AI Public Multicenter Validation Release v3.3.5

## Summary

CardioTwin-AI is a research-use ECG intelligence platform for 12-lead ECG preliminary screening, visual explanation, and evidence-based referral support.

This repository contains the frozen public validation release v3.3.5, including source code, Streamlit dashboards, validation scripts, selected reports, final evidence pack, model checkpoint, threshold profile, and claim-boundary documentation.

## Current Validation Status

- Locked public multicenter cohort: 3,222 ECG records
- Sources: CPSC 2018, CPSC Extra, PTB, Georgia
- Frozen runtime inference: 3,222 / 3,222 completed
- Error count: 0
- CPU runtime: approximately 319.84 seconds
- Final report: artifacts/public_multicenter_validation_v33/PUBLIC_MULTICENTER_VALIDATION_FINAL_REPORT_v335.md

## Key Metrics

Source-separated macro AUROC:

- PTB: 0.9164
- CPSC 2018: 0.8394
- CPSC Extra: 0.8076
- Georgia: 0.7546
- All sources stacked reference only: 0.8090

The stacked result is descriptive only and must not be interpreted as a random-split validation claim.

## Main Streamlit Commands

### Core dashboard

Run this from PowerShell:

    cd C:\Users\mrkit\Downloads\cardiotwin-ai-public-validation-v335

    $PY = "C:\venvs\cardiotwin_v25\Scripts\python.exe"
    $env:PYTHONPATH = "$PWD\src"
    $env:MPLBACKEND = "Agg"

    & $PY -m streamlit run apps\streamlit_dashboard_v27_export_pack.py --server.port 8507

Open:

    http://localhost:8507

### Unified dashboard, if available

Run this from PowerShell:

    cd C:\Users\mrkit\Downloads\cardiotwin-ai-public-validation-v335

    $PY = "C:\venvs\cardiotwin_v25\Scripts\python.exe"
    $env:PYTHONPATH = "$PWD\src"
    $env:MPLBACKEND = "Agg"

    $unified = @(
      "apps\streamlit_cardiotwin_unified_v304.py",
      "apps\streamlit_cardiotwin_unified_v303.py",
      "apps\streamlit_cardiotwin_unified_v302.py",
      "apps\streamlit_cardiotwin_unified_v304_real_inference.py"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($null -eq $unified) {
      Write-Host "No unified dashboard found. Available Streamlit apps:"
      dir apps\*streamlit*.py
    } else {
      Write-Host "Running unified dashboard:" $unified
      & $PY -m streamlit run $unified --server.port 8508
    }

Open:

    http://localhost:8508

## Important Reports

- Final report: artifacts/public_multicenter_validation_v33/PUBLIC_MULTICENTER_VALIDATION_FINAL_REPORT_v335.md
- Final HTML report: artifacts/public_multicenter_validation_v33/public_multicenter_validation_final_report_v335.html
- Paper-ready results table: artifacts/public_multicenter_validation_v33/PUBLIC_PAPER_READY_RESULTS_TABLE_v335.csv
- Claim boundary: artifacts/public_multicenter_validation_v33/PUBLIC_CLAIM_BOUNDARY_AND_NEXT_STEPS_v335.md
- Doctor review template: artifacts/public_multicenter_validation_v33/doctor_in_the_loop_review_template_v334.csv
- Release pack: artifacts/release_rc1/cardiotwin_v3_3_5_final_public_multicenter_validation_pack.zip
- Release manifest: artifacts/release_rc1/cardiotwin_v3_3_5_final_public_multicenter_validation_manifest.json

## What This System Can Do Now

- Load and analyze 12-lead ECG records.
- Run frozen InceptionTime ECG AI inference.
- Output probabilities for NORM, MI, STTC, CD, and HYP.
- Apply safety threshold profiles.
- Provide preliminary screening flags.
- Support visual explanation and region-mapping workflow where available.
- Export evidence reports.
- Provide failure-case review and doctor-in-the-loop template.

## Claim Boundary

This project is for research-use public multicenter validation evidence only.

Do not claim yet:

- clinically validated
- doctor-level diagnosis
- ready for autonomous clinical diagnosis
- generalizes to every hospital
- prospectively validated

Safe claim:

CardioTwin-AI was evaluated with a frozen runtime on a locked, source-separated public multi-center ECG cohort. The system showed promising discrimination across multiple public sources, with screening-oriented sensitivity and a need for source-aware threshold calibration and doctor-in-the-loop review before clinical claims.

## Data Policy

Raw ECG datasets are intentionally excluded from this repository.

Do not push:

- data/raw/
- credentialed or restricted medical datasets
- patient-identifiable data
- passwords, tokens, or private credentials

## Next Roadmap

- v3.4: Expanded public source-aware calibration
- v3.5: Doctor-in-the-loop adjudication report
- v3.6: Prospective pilot protocol and safety/risk pack
