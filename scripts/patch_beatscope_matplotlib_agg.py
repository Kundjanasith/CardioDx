from pathlib import Path

p = Path("scripts/run_beatscope_benchmark_v28.py")
text = p.read_text(encoding="utf-8")

old = "import joblib\nimport matplotlib.pyplot as plt"
new = "import joblib\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"

if old in text and "matplotlib.use('Agg')" not in text:
    text = text.replace(old, new)

p.write_text(text, encoding="utf-8")
print("Patched matplotlib backend to Agg.")
