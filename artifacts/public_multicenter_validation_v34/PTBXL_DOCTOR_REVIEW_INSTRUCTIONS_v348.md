# CardioTwin-AI v3.4.8 PTB-XL Doctor-review Template Instructions

Created: 2026-05-31T11:52:47.493720+00:00

## Purpose

This template prepares PTB-XL fold 10 failure cases for structured expert review.

## Reviewer Task

For each review_case_id, inspect the ECG record and complete the reviewer fields.

Recommended fields to complete:

- reviewer_id
- review_date
- ecg_quality_review
- label_present_by_reviewer
- reviewer_primary_label
- reviewer_secondary_labels
- review_confidence_1_to_5
- ai_error_category
- recommended_action
- needs_second_reviewer
- reviewer_comments

## Review Policy Meaning

- runtime_screening: high-sensitivity screening threshold.
- fold9_best_f1: fold-9-derived threshold intended to reduce false positives and improve F1/specificity.

## Claim Boundary

This is a structured expert-review template. It is not a clinical diagnosis, not prospective validation, and not clinical deployment.