from pathlib import Path

path = Path("scripts/evaluate_cinc2020_georgia_external.py")
text = path.read_text(encoding="utf-8")

new_func = '''def build_code_to_class(cinc_root: Path) -> dict[str, str]:
    """
    Build SNOMED -> PTB-XL superclass mapping.

    Priority:
    1. configs/cinc2020_to_ptbxl_superclass_map_v21.csv
    2. configs/cinc2020_to_ptbxl_superclass_map.csv
    3. fallback direct mapping inside this script
    """
    config_candidates = [
        Path("configs/cinc2020_to_ptbxl_superclass_map_v21.csv"),
        Path("configs/cinc2020_to_ptbxl_superclass_map.csv"),
    ]

    for cfg in config_candidates:
        if cfg.exists():
            df = pd.read_csv(cfg)
            df["code"] = df["code"].astype(str)

            mapping = {}
            for _, row in df.iterrows():
                decision = str(row.get("decision", "")).strip()
                cls = str(row.get("ptbxl_superclass", "")).strip()
                code = str(row.get("code", "")).strip()

                if decision == "include" and cls in PTBXL_SUPERCLASSES and code:
                    mapping[code] = cls

            print(f"[INFO] Loaded harmonization map: {cfg} | mapped_codes={len(mapping)}")
            return mapping

    # Fallback only if no harmonization CSV is available.
    mapping = dict(DIRECT_CODE_TO_CLASS)

    for fname in ["dx_mapping_scored.csv", "dx_mapping_unscored.csv"]:
        path = cinc_root / fname
        if not path.exists():
            continue

        df = pd.read_csv(path)
        lower = {c.lower().strip(): c for c in df.columns}

        code_col = None
        for key in ["snomed ct code", "snomed_ct_code", "snomedctcode", "code"]:
            if key in lower:
                code_col = lower[key]
                break

        if code_col is None:
            continue

        text_cols = []
        for key in ["diagnosis", "abbreviation", "description", "dx"]:
            if key in lower:
                text_cols.append(lower[key])

        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            text = " ".join(str(row[c]) for c in text_cols if c in row.index)

            for cls, pat in PATTERNS:
                if pat.search(text):
                    mapping.setdefault(code, cls)
                    break

    print(f"[INFO] Loaded fallback direct mapping | mapped_codes={len(mapping)}")
    return mapping
'''

start = text.find("def build_code_to_class(")
if start == -1:
    raise RuntimeError("Could not find def build_code_to_class")

end = text.find("\ndef parse_dx", start)
if end == -1:
    raise RuntimeError("Could not find def parse_dx after build_code_to_class")

text2 = text[:start] + new_func + "\n\n" + text[end + 1:]
path.write_text(text2, encoding="utf-8")
print("Patched evaluator to use harmonization CSV.")
