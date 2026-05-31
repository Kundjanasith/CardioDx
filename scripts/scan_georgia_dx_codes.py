from pathlib import Path
from collections import Counter
import pandas as pd

root = Path("data/raw/cinc2020/training/georgia")
headers = sorted(root.rglob("*.hea"))

counter = Counter()
examples = {}

for h in headers:
    text = h.read_text(errors="ignore", encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("#Dx:"):
            codes = [x.strip() for x in line.replace("#Dx:", "").split(",") if x.strip()]
            for c in codes:
                counter[c] += 1
                examples.setdefault(c, h.name)
            break

rows = []
for code, count in counter.most_common(100):
    rows.append({
        "code": code,
        "count": count,
        "example_file": examples.get(code, "")
    })

out = Path("artifacts/external_validation/georgia_true_eval")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "georgia_top_dx_codes.csv", index=False)

print("headers:", len(headers))
print("unique dx codes:", len(counter))
print(pd.DataFrame(rows).head(30).to_string(index=False))
print("Saved:", out / "georgia_top_dx_codes.csv")
