from pathlib import Path
import pandas as pd

p = Path("artifacts/public_multicenter_validation_v33/public_locked_validation_cohort_v330.csv")
df = pd.read_csv(p)

print("rows:", len(df))
print("waveform_ready counts:")
print(df["waveform_ready"].value_counts(dropna=False))

print("\nby source:")
print(df.groupby("source_id")["waveform_ready"].value_counts(dropna=False))

print("\nmissing waveform examples:")
print(df[df["waveform_ready"] == False][["source_id", "record_id", "hea_path", "mat_path", "dat_path"]].head(20).to_string(index=False))
