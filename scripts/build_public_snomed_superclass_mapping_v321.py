from pathlib import Path
import csv
import json
import urllib.request
from datetime import datetime, timezone
from collections import defaultdict, Counter

OUT = Path("artifacts/public_multicenter_validation_v32")
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "dx_mapping_scored.csv": "https://raw.githubusercontent.com/physionetchallenges/evaluation-2020/master/dx_mapping_scored.csv",
    "dx_mapping_unscored.csv": "https://raw.githubusercontent.com/physionetchallenges/evaluation-2020/master/dx_mapping_unscored.csv",
}

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def download_if_missing(name, url):
    p = OUT / name
    if p.exists() and p.stat().st_size > 0:
        return p, "already_exists"

    try:
        print("[DOWNLOAD]", url)
        urllib.request.urlretrieve(url, p)
        return p, "downloaded"
    except Exception as e:
        return p, "download_failed:" + repr(e)


def s(x):
    return str(x or "").strip()


def infer_superclass(dx, abbr):
    d = s(dx).lower()
    a = s(abbr).upper()
    labels = set()

    if a in {"NORM", "NSR", "NORMAL", "SR"}:
        labels.add("NORM")
    if "normal ecg" in d or "normal sinus rhythm" in d or d == "normal" or d == "sinus rhythm":
        labels.add("NORM")

    if a in {"MI", "AMI", "OLDMI", "IMI", "LMI", "PMI"}:
        labels.add("MI")
    if "myocardial infarction" in d or "infarction" in d or "infarct" in d:
        labels.add("MI")

    if a in {
        "NSSTTA", "STD", "STE", "STC", "STIAB", "TAB", "TINV",
        "MIS", "AMIS", "CMI", "IIS", "LIS", "ERRE", "ERE",
        "NDT", "NST_", "ISC_"
    }:
        labels.add("STTC")
    if any(k in d for k in [
        "st depression", "st elevation", "st segment", "s-t segment",
        "st-t", "nonspecific st", "non-specific st", "st t abnormality",
        "t wave abnormal", "t-wave abnormal", "t wave inversion",
        "ischemia", "ischaemia", "repolarization", "repolarisation"
    ]):
        labels.add("STTC")

    if a in {
        "IAVB", "IIAVB", "AVB", "CHB", "RBBB", "CRBBB", "IRBBB",
        "LBBB", "ILBBB", "LAFB", "LANFB", "LPFB", "BBB",
        "NSIVCB", "DIB", "LPR", "IVCD"
    }:
        labels.add("CD")
    if any(k in d for k in [
        "av block", "atrioventricular block", "bundle branch block",
        "right bundle branch block", "left bundle branch block",
        "conduction", "fascicular block", "intraventricular block",
        "hemiblock", "prolonged pr"
    ]):
        labels.add("CD")

    if a in {"LVH", "RVH", "VH"}:
        labels.add("HYP")
    if "hypertrophy" in d:
        labels.add("HYP")

    return sorted(labels)


def read_mapping_file(path, mapping_kind):
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dx = s(r.get("Dx"))
            code = s(r.get("SNOMED CT Code"))
            abbr = s(r.get("Abbreviation"))
            labels = infer_superclass(dx, abbr)
            rows.append({
                "mapping_kind": mapping_kind,
                "dx": dx,
                "snomed_code": code,
                "abbreviation": abbr,
                "target_superclasses": "|".join(labels),
                "is_mapped_to_target": bool(labels),
            })
    return rows


def main():
    created = datetime.now(timezone.utc).isoformat()
    downloaded = {}
    all_rows = []

    for name, url in URLS.items():
        p, status = download_if_missing(name, url)
        downloaded[name] = {
            "path": str(p),
            "status": status,
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
        }

        if p.exists() and p.stat().st_size > 0:
            kind = "scored" if name == "dx_mapping_scored.csv" else "unscored"
            all_rows.extend(read_mapping_file(p, kind))

    if not all_rows:
        raise RuntimeError("No mapping rows loaded. Download failed or files are missing.")

    out_csv = OUT / "public_snomed_to_superclass_mapping_v321.csv"
    discoverable = OUT / "dx_mapping_public_superclass_v321.csv"
    out_json = OUT / "public_snomed_to_superclass_mapping_v321.json"
    md = OUT / "PUBLIC_SNOMED_SUPERCLASS_MAPPING_v321.md"

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "mapping_kind",
            "dx",
            "snomed_code",
            "abbreviation",
            "target_superclasses",
            "is_mapped_to_target",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    with discoverable.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "Dx",
            "SNOMED CT Code",
            "Abbreviation",
            "target_superclasses",
            "mapping_kind",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "Dx": r["dx"],
                "SNOMED CT Code": r["snomed_code"],
                "Abbreviation": r["abbreviation"],
                "target_superclasses": r["target_superclasses"],
                "mapping_kind": r["mapping_kind"],
            })

    support = Counter()
    code_to_labels = defaultdict(set)

    for r in all_rows:
        labs = [x for x in r["target_superclasses"].split("|") if x]
        for lab in labs:
            support[lab] += 1
            if r["snomed_code"]:
                code_to_labels[r["snomed_code"]].add(lab)

    summary = {
        "project": "CardioTwin-AI",
        "version": "v3.2-public.1 official SNOMED superclass mapping",
        "created_at_utc": created,
        "source_files": downloaded,
        "n_mapping_rows": len(all_rows),
        "n_codes_mapped_to_target": len(code_to_labels),
        "target_classes": TARGET_CLASSES,
        "support_by_target_class": {k: int(support[k]) for k in TARGET_CLASSES},
        "outputs": {
            "public_snomed_to_superclass_mapping_csv": str(out_csv),
            "discoverable_dx_mapping_csv": str(discoverable),
            "summary_json": str(out_json),
            "summary_md": str(md),
        },
        "important_boundary": "This is label harmonization for research validation. Metrics must be reported by source and not pooled as a random split.",
    }

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md.write_text(
        "# CardioTwin-AI v3.2-public.1 Official SNOMED Superclass Mapping\n\n"
        f"Created: {created}\n\n"
        "## Purpose\n\n"
        "Fix public multi-center label harmonization by importing official PhysioNet/CinC Challenge diagnosis mapping files and mapping SNOMED diagnosis terms into CardioTwin-AI target superclasses.\n\n"
        "## Target Superclasses\n\n"
        "- NORM\n"
        "- MI\n"
        "- STTC\n"
        "- CD\n"
        "- HYP\n\n"
        "## Outputs\n\n"
        f"- {out_csv}\n"
        f"- {discoverable}\n"
        f"- {out_json}\n\n"
        "## Summary JSON\n\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n\n## Claim Boundary\n\n"
        "This is label harmonization for research validation. Metrics must still be reported by source and not pooled as a random split.\n",
        encoding="utf-8",
    )

    print("DONE: v3.2-public.1 official SNOMED superclass mapping")
    print("CSV:", out_csv)
    print("DISCOVERABLE:", discoverable)
    print("JSON:", out_json)
    print("MD:", md)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
