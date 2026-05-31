from pathlib import Path
import pandas as pd

root = Path("data/raw/cinc2020")
eval_dir = Path("artifacts/external_validation/georgia_true_eval")

coverage = pd.read_csv(eval_dir / "georgia_dx_code_coverage.csv")
coverage["code"] = coverage["code"].astype(str)

maps = []
for name in ["dx_mapping_scored.csv", "dx_mapping_unscored.csv"]:
    p = root / name
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        df["source_file"] = name
        maps.append(df)

if maps:
    m = pd.concat(maps, ignore_index=True)
    lower = {c.lower(): c for c in m.columns}
    code_col = None
    for c in ["snomed ct code", "snomed_ct_code", "snomedctcode", "code"]:
        if c in lower:
            code_col = lower[c]
            break

    if code_col is None:
        raise RuntimeError(f"Cannot find SNOMED code column in mapping files: {m.columns.tolist()}")

    m["code"] = m[code_col].astype(str)

    keep_cols = ["code", "source_file"]
    for c in ["Abbreviation", "abbreviation", "Diagnosis", "diagnosis", "Description", "description"]:
        if c in m.columns and c not in keep_cols:
            keep_cols.append(c)

    joined = coverage.merge(m[keep_cols], on="code", how="left")
else:
    joined = coverage.copy()
    joined["source_file"] = ""

unmapped = joined[joined["mapped_class"].fillna("") == ""].copy()
unmapped["count"] = unmapped["count"].astype(int)
unmapped = unmapped.sort_values("count", ascending=False)

out = eval_dir / "georgia_unmapped_codes_with_text.csv"
unmapped.to_csv(out, index=False)

print(unmapped.head(50).to_string(index=False))
print("Saved:", out)
