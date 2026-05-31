from pathlib import Path
import importlib.util

script_path = Path("scripts/evaluate_cinc2020_georgia_external.py")

spec = importlib.util.spec_from_file_location("geo_eval", script_path)
geo_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo_eval)

h = Path("data/raw/cinc2020/training/georgia/g1/E00001.hea")
print(geo_eval.parse_dx(h))
