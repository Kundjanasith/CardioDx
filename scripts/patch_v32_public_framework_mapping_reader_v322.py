from pathlib import Path
import re

path = Path("scripts/build_v32_public_multicenter_framework.py")

if not path.exists():
    raise FileNotFoundError(path)

text = path.read_text(encoding="utf-8")

new_func = r'''def load_code_mapping():
    code_to_super = defaultdict(set)
    mapping_sources = []

    # Seed mapping kept for safety.
    for code, labels in SEED_CODE_TO_SUPER.items():
        for label in labels:
            if label in TARGET_CLASSES:
                code_to_super[str(code)].add(label)

    # Highest-priority project harmonization files, if present.
    # These were used successfully by earlier external validation scripts.
    config_candidates = [
        Path("configs/cinc2020_to_ptbxl_superclass_map_v21.csv"),
        Path("configs/cinc2020_to_ptbxl_superclass_map.csv"),
    ]

    for cfg in config_candidates:
        if cfg.exists():
            try:
                with cfg.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    reader = csv.DictReader(f)
                    cols = reader.fieldnames or []
                    lower = {c.lower().strip(): c for c in cols}

                    code_col = lower.get("code") or lower.get("snomed_code") or lower.get("snomed ct code")
                    decision_col = lower.get("decision")
                    cls_col = lower.get("ptbxl_superclass") or lower.get("target_superclass") or lower.get("superclass")

                    if code_col and cls_col:
                        for row in reader:
                            code = str(row.get(code_col, "")).strip()
                            cls = str(row.get(cls_col, "")).strip()

                            decision = "include"
                            if decision_col:
                                decision = str(row.get(decision_col, "")).strip().lower()

                            if code and cls in TARGET_CLASSES and decision == "include":
                                code_to_super[code].add(cls)

                mapping_sources.append(str(cfg))
            except Exception as e:
                mapping_sources.append(f"{cfg}:error:{repr(e)}")

    # Public SNOMED mapping files and official dx_mapping files.
    for p in find_dx_mapping_files():
        try:
            with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                lower = {c.lower().strip(): c for c in cols}

                code_col = None
                for key in [
                    "snomed ct code",
                    "snomed_ct_code",
                    "snomedctcode",
                    "snomed code",
                    "code",
                    "dx",
                    "diagnosis_code",
                ]:
                    if key in lower:
                        code_col = lower[key]
                        break

                target_col = None
                for key in [
                    "target_superclasses",
                    "target_superclass",
                    "ptbxl_superclass",
                    "superclass",
                ]:
                    if key in lower:
                        target_col = lower[key]
                        break

                term_cols = []
                for key in [
                    "dx",
                    "diagnosis",
                    "description",
                    "term",
                    "diagnostic_class",
                    "diagnostic_subclass",
                ]:
                    if key in lower:
                        term_cols.append(lower[key])

                abbr_col = None
                for key in ["abbreviation", "abbr"]:
                    if key in lower:
                        abbr_col = lower[key]
                        break

                for row in reader:
                    if not code_col:
                        continue

                    code = str(row.get(code_col, "")).strip()
                    if not code:
                        continue

                    labels = set()

                    # Preferred path: use frozen/generated target_superclasses directly.
                    if target_col:
                        raw = str(row.get(target_col, "")).strip()
                        for lab in re.split(r"[|,;/ ]+", raw):
                            lab = lab.strip()
                            if lab in TARGET_CLASSES:
                                labels.add(lab)

                    # Fallback: infer from diagnosis text and abbreviation.
                    if not labels:
                        term = " ".join(str(row.get(c, "")) for c in term_cols)
                        abbr = str(row.get(abbr_col, "")) if abbr_col else ""
                        for lab in infer_super_from_term(term, abbr):
                            if lab in TARGET_CLASSES:
                                labels.add(lab)

                    for lab in labels:
                        code_to_super[code].add(lab)

            mapping_sources.append(str(p))

        except Exception as e:
            mapping_sources.append(f"{p}:error:{repr(e)}")
            continue

    return {k: sorted(v) for k, v in code_to_super.items()}, mapping_sources
'''

pattern = r"def load_code_mapping\(\):.*?\n(?=def parse_hea_header\(path\):)"

new_text, n = re.subn(pattern, new_func + "\n\n", text, flags=re.DOTALL)

if n != 1:
    raise RuntimeError(f"Could not patch load_code_mapping cleanly. replacements={n}")

backup = path.with_suffix(".py.v32_public_mapping_reader_backup")
backup.write_text(text, encoding="utf-8")
path.write_text(new_text, encoding="utf-8")

print("DONE: patched build_v32_public_multicenter_framework.py")
print("backup:", backup)
print("replacements:", n)
