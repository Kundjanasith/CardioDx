from pathlib import Path
import requests

base = "https://physionet.org/files/challenge-2020/1.0.2/"
out = Path("data/raw/cinc2020")
out.mkdir(parents=True, exist_ok=True)

files = [
    "dx_mapping_scored.csv",
    "dx_mapping_unscored.csv",
    "weights.csv",
    "README.md",
]

for f in files:
    url = base + f
    path = out / f
    print("Downloading", url)
    r = requests.get(url, timeout=120)
    if r.status_code == 200:
        path.write_bytes(r.content)
        print("[OK]", path)
    else:
        print("[SKIP]", f, r.status_code)
