from pathlib import Path
import pandas as pd
import json

path = Path("configs/cinc2020_to_ptbxl_superclass_map.csv")
df = pd.read_csv(path)
df["code"] = df["code"].astype(str)

corrections = {
    # Fix scientifically suspicious inherited mappings
    "55930002": {
        "ptbxl_superclass": "STTC",
        "decision": "include",
        "notes": "manual_v21_fix_st_changes_to_STTC",
    },
    "39732003": {
        "ptbxl_superclass": "",
        "decision": "exclude_other",
        "notes": "manual_v21_fix_left_axis_deviation_not_5_superclass",
    },
    "111975006": {
        "ptbxl_superclass": "STTC",
        "decision": "include",
        "notes": "manual_v21_fix_prolonged_qt_repolarization_proxy_STTC",
    },

    # Review decisions
    "251268003": {
        "ptbxl_superclass": "",
        "decision": "exclude_other",
        "notes": "manual_v21_atrial_pacing_pattern_not_5_superclass",
    },
    "74390002": {
        "ptbxl_superclass": "CD",
        "decision": "include",
        "notes": "manual_v21_wpw_preexcitation_conduction_proxy",
    },
    "195060002": {
        "ptbxl_superclass": "CD",
        "decision": "include",
        "notes": "manual_v21_ventricular_pre_excitation_conduction_proxy",
    },
}

for code, vals in corrections.items():
    mask = df["code"] == code
    if not mask.any():
        continue
    for k, v in vals.items():
        df.loc[mask, k] = v

out_path = Path("configs/cinc2020_to_ptbxl_superclass_map_v21.csv")
df.to_csv(out_path, index=False)

summary = {
    "map_file": str(out_path),
    "n_codes": int(len(df)),
    "n_include_codes": int((df["decision"] == "include").sum()),
    "n_exclude_other_codes": int((df["decision"] == "exclude_other").sum()),
    "n_review_codes": int((df["decision"] == "review").sum()),
    "include_record_weight_by_code_count": int(df[df["decision"] == "include"]["count"].sum()),
    "exclude_other_record_weight_by_code_count": int(df[df["decision"] == "exclude_other"]["count"].sum()),
    "review_record_weight_by_code_count": int(df[df["decision"] == "review"]["count"].sum()),
    "manual_corrections": corrections,
}

out_summary = Path("artifacts/external_validation/georgia_true_eval/harmonization_v21_summary.json")
out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("Saved:", out_path)
