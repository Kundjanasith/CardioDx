from pathlib import Path
import requests
import pandas as pd

url = "https://raw.githubusercontent.com/physionetchallenges/physionetchallenges.github.io/master/2020/Dx_map.csv"

out_dir = Path("data/raw/cinc2020")
out_dir.mkdir(parents=True, exist_ok=True)

raw_path = out_dir / "Dx_map_raw.csv"
clean_path = out_dir / "Dx_map_clean.csv"

print("Downloading:", url)
r = requests.get(url, timeout=120)
r.raise_for_status()
raw_path.write_bytes(r.content)

text = r.text.strip()

# The raw file may appear as one long comma-separated stream:
# Dx,SNOMED CT Code,Abbreviation,diagnosis,code,abbr,...
parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]

if len(parts) < 6:
    raise RuntimeError("Dx_map content too short or unexpected.")

# Remove header triplet.
if parts[0].lower() == "dx":
    parts = parts[3:]

rows = []
for i in range(0, len(parts) - 2, 3):
    rows.append({
        "diagnosis": parts[i],
        "code": str(parts[i + 1]),
        "abbreviation": parts[i + 2],
    })

df = pd.DataFrame(rows)
df.to_csv(clean_path, index=False)

print(df.head(20).to_string(index=False))
print("Saved:", clean_path)
print("n_rows:", len(df))
