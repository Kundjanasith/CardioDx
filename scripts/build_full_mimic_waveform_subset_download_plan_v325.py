from pathlib import Path
import csv
import json
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(".")
OUT = Path("artifacts/label_supported_external_validation_v32")
OUT.mkdir(parents=True, exist_ok=True)

SUBSET_PLAN = OUT / "full_mimic_waveform_subset_plan_v324.csv"
RAW_ROOT = Path("data/raw/mimic_iv_ecg")
BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0/"
PER_LABEL = 20

def clean_path(p):
    s = str(p or "").strip().replace("\\", "/")
    s = s.replace("data/raw/mimic_iv_ecg/", "")
    s = s.replace("data\\raw\\mimic_iv_ecg\\", "")
    s = s.lstrip("./")
    if s.endswith(".hea"):
        s = s[:-4]
    if s.endswith(".dat"):
        s = s[:-4]
    return s

if not SUBSET_PLAN.exists():
    raise FileNotFoundError(f"Missing subset plan: {SUBSET_PLAN}")

rows = []
with SUBSET_PLAN.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

selected = []
seen_base = set()
label_counts = defaultdict(int)

for r in rows:
    label = str(r.get("target_label", "")).strip()
    if not label:
        continue

    if label_counts[label] >= PER_LABEL:
        continue

    base = clean_path(r.get("path", ""))
    if not base:
        base = clean_path(r.get("hea_path_guess", ""))

    if not base or base in seen_base:
        continue

    seen_base.add(base)
    label_counts[label] += 1

    hea_rel = base + ".hea"
    dat_rel = base + ".dat"
    hea_local = RAW_ROOT / hea_rel
    dat_local = RAW_ROOT / dat_rel

    out = dict(r)
    out.update({
        "record_base_rel": base,
        "hea_rel": hea_rel,
        "dat_rel": dat_rel,
        "hea_url": BASE_URL + hea_rel,
        "dat_url": BASE_URL + dat_rel,
        "hea_local": str(hea_local),
        "dat_local": str(dat_local),
        "hea_exists_now": hea_local.exists(),
        "dat_exists_now": dat_local.exists(),
        "ready_pair": hea_local.exists() and dat_local.exists(),
    })
    selected.append(out)

manifest_csv = OUT / "full_mimic_waveform_subset_download_manifest_v325.csv"
urls_txt = OUT / "full_mimic_waveform_subset_download_urls_v325.txt"
readiness_json = OUT / "full_mimic_waveform_subset_readiness_v325.json"
commands_txt = OUT / "FULL_MIMIC_WAVEFORM_SUBSET_DOWNLOAD_COMMANDS_v325.txt"

fieldnames = []
for r in selected:
    for k in r.keys():
        if k not in fieldnames:
            fieldnames.append(k)

with manifest_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(selected)

urls = []
for r in selected:
    urls.append(r["hea_url"])
    urls.append(r["dat_url"])

urls_txt.write_text("\n".join(urls) + "\n", encoding="utf-8")

ready_pairs = sum(1 for r in selected if r["ready_pair"])
missing_pairs = len(selected) - ready_pairs

ready_by_label = defaultdict(int)
for r in selected:
    if r["ready_pair"]:
        ready_by_label[r.get("target_label", "")] += 1

payload = {
    "project": "CardioTwin-AI",
    "version": "v3.2.5 Full MIMIC-IV-ECG waveform subset download plan",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_subset_plan": str(SUBSET_PLAN),
    "per_label": PER_LABEL,
    "selected_record_pairs": len(selected),
    "selected_files_to_download": len(urls),
    "ready_record_pairs": ready_pairs,
    "missing_record_pairs": missing_pairs,
    "label_counts": dict(label_counts),
    "ready_by_label": dict(ready_by_label),
    "readiness": "ready_for_subset_inference" if len(selected) > 0 and missing_pairs == 0 else "download_required",
    "outputs": {
        "download_manifest_csv": str(manifest_csv),
        "download_urls_txt": str(urls_txt),
        "download_commands_txt": str(commands_txt),
    },
    "claim_boundary": "Download planning only. No model validation metrics computed."
}

readiness_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

commands_txt.write_text(
    "Run these commands after confirming wget.exe is available.\n\n"
    "cd C:\\Users\\mrkit\\Downloads\\cardiotwin_ai_12l\\data\\raw\\mimic_iv_ecg\n\n"
    "wget.exe -N -c --user YOUR_PHYSIONET_USERNAME --ask-password -x -nH --cut-dirs=3 -i C:\\Users\\mrkit\\Downloads\\cardiotwin_ai_12l\\artifacts\\label_supported_external_validation_v32\\full_mimic_waveform_subset_download_urls_v325.txt\n\n"
    "Do not paste your password into chat or save it in scripts.\n",
    encoding="utf-8"
)

print("DONE: v3.2.5 waveform subset download plan")
print("MANIFEST_CSV:", manifest_csv)
print("URLS_TXT:", urls_txt)
print("READINESS_JSON:", readiness_json)
print("COMMANDS_TXT:", commands_txt)
print(json.dumps(payload, indent=2, ensure_ascii=False))
