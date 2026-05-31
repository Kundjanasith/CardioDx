LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PTBXL_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
REGIONS = ["inferior", "septal", "anterior", "lateral", "global_conduction", "global_rhythm", "uncertain"]
LEAD_TO_REGION = {
    "II": "inferior", "III": "inferior", "aVF": "inferior",
    "V1": "septal", "V2": "septal",
    "V3": "anterior", "V4": "anterior",
    "I": "lateral", "aVL": "lateral", "V5": "lateral", "V6": "lateral",
}
REGION_TO_LEADS = {
    region: [lead for lead, r in LEAD_TO_REGION.items() if r == region]
    for region in sorted(set(LEAD_TO_REGION.values()))
}
CLASS_TO_REGION_PRIOR = {
    "MI": {"inferior": 0.28, "anterior": 0.28, "septal": 0.20, "lateral": 0.24},
    "STTC": {"inferior": 0.25, "anterior": 0.25, "septal": 0.20, "lateral": 0.30},
    "CD": {"global_conduction": 0.75, "septal": 0.10, "anterior": 0.05, "lateral": 0.05, "inferior": 0.05},
    "HYP": {"lateral": 0.35, "anterior": 0.25, "septal": 0.20, "inferior": 0.10, "global_conduction": 0.10},
    "NORM": {"inferior": 0.0, "anterior": 0.0, "septal": 0.0, "lateral": 0.0, "global_conduction": 0.0},
}
