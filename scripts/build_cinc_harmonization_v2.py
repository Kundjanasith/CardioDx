from pathlib import Path
import pandas as pd
import json

eval_dir = Path("artifacts/external_validation/georgia_true_eval")
root = Path("data/raw/cinc2020")

coverage = pd.read_csv(eval_dir / "georgia_dx_code_coverage.csv")
coverage["code"] = coverage["code"].astype(str)

dx = pd.read_csv(root / "Dx_map_clean.csv")
dx["code"] = dx["code"].astype(str)

joined = coverage.merge(dx, on="code", how="left")
joined["diagnosis"] = joined["diagnosis"].fillna("")
joined["abbreviation"] = joined["abbreviation"].fillna("")
joined["mapped_class"] = joined["mapped_class"].fillna("")

def classify(row):
    code = str(row["code"])
    dx_text = str(row["diagnosis"]).lower()
    abbr = str(row["abbreviation"]).upper()

    # Already mapped by previous evaluator: keep it.
    existing = str(row.get("mapped_class", "")).strip()
    if existing in {"NORM", "MI", "STTC", "CD", "HYP"}:
        return existing, "include", "existing_mapping"

    # Normal / sinus rhythm
    if code in {"426783006", "426177001"} or abbr in {"NSR", "SB"}:
        return "NORM", "include", "normal_or_sinus_rhythm"

    # ST/T / ischemia / repolarization
    if (
        "ischemia" in dx_text or "ischaemia" in dx_text or
        "st " in dx_text or dx_text.startswith("st ") or
        "t wave" in dx_text or "repolarization" in dx_text or
        abbr in {"NSSTTA", "TAB", "TINV", "STD", "STE", "STIAB", "STC", "ANMIS", "IIS", "LIS", "ERЕ", "ERE"}
    ):
        return "STTC", "include", "ischemia_or_st_t_mapping"

    # MI / infarction
    if "myocardial infarction" in dx_text or "infarction" in dx_text or abbr in {"MI", "AMI", "OLDMI", "ANMI"}:
        return "MI", "include", "infarction_mapping"

    # Conduction
    if (
        "block" in dx_text or "conduction" in dx_text or
        "bundle branch" in dx_text or "fascicular" in dx_text or
        abbr in {"IAVB", "IIAVB", "AVB", "BBB", "LBBB", "RBBB", "IRBBB", "ILBBB", "CRBBB", "LAFB", "LPFB", "NSIVCB", "CHB"}
    ):
        return "CD", "include", "conduction_mapping"

    # Hypertrophy / chamber enlargement/abnormality
    if (
        "hypertrophy" in dx_text or "enlargement" in dx_text or
        "atrial abnormality" in dx_text or
        abbr in {"LVH", "RVH", "VH", "LAE", "LAA", "RAAB", "AH", "RAH", "LAH"}
    ):
        return "HYP", "include", "hypertrophy_or_chamber_abnormality_mapping"

    # Rhythm classes: keep for future 6th/7th class, do not force into 5-class.
    rhythm_words = [
        "atrial fibrillation", "atrial flutter", "tachycardia", "bradycardia",
        "premature", "ectopic", "escape", "junctional", "bigeminy", "trigeminy",
        "fibrillation", "flutter", "wandering atrial", "pacing rhythm", "sinus arrhythmia"
    ]
    if any(w in dx_text for w in rhythm_words) or abbr in {
        "AF", "AFL", "AFAFL", "PAC", "PVC", "VPB", "VEB", "SVT", "VTACH",
        "STACH", "BRADY", "SA", "PAF", "AJR", "WAP", "VPP", "PR"
    }:
        return "", "exclude_other", "rhythm_not_in_5_superclass"

    # Axis / voltage / artifact / nonspecific form
    if abbr in {"RAD", "LAD", "ICA", "LQRSV", "ALR", "QAB", "RAB", "SPRI", "SQT", "HTV"}:
        return "", "exclude_other", "axis_voltage_form_or_artifact_not_in_5_superclass"

    return "", "review", "needs_manual_review"

rows = []
for _, row in joined.iterrows():
    cls, decision, notes = classify(row)
    rows.append({
        "code": str(row["code"]),
        "count": int(row["count"]),
        "diagnosis": row["diagnosis"],
        "abbreviation": row["abbreviation"],
        "previous_mapped_class": row.get("mapped_class", ""),
        "ptbxl_superclass": cls,
        "decision": decision,
        "notes": notes,
    })

harm = pd.DataFrame(rows).sort_values("count", ascending=False)
out = Path("configs")
out.mkdir(exist_ok=True)
harm_path = out / "cinc2020_to_ptbxl_superclass_map.csv"
harm.to_csv(harm_path, index=False)

summary = {
    "n_codes": int(len(harm)),
    "n_include_codes": int((harm["decision"] == "include").sum()),
    "n_exclude_other_codes": int((harm["decision"] == "exclude_other").sum()),
    "n_review_codes": int((harm["decision"] == "review").sum()),
    "include_record_weight_by_code_count": int(harm[harm["decision"] == "include"]["count"].sum()),
    "exclude_other_record_weight_by_code_count": int(harm[harm["decision"] == "exclude_other"]["count"].sum()),
    "review_record_weight_by_code_count": int(harm[harm["decision"] == "review"]["count"].sum()),
}

(eval_dir / "harmonization_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(harm.head(60).to_string(index=False))
print("Saved:", harm_path)
print(json.dumps(summary, indent=2))
