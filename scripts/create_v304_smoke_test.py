from pathlib import Path

script = r'''
from pathlib import Path
import json
from datetime import datetime, timezone

from cardiotwin.runtime.v304_real_inference_bridge import (
    load_wfdb_hea_mat,
    synthetic_ecg,
    run_v304_real_inference,
)

OUT = Path("artifacts/unified_demo_v304")
OUT.mkdir(parents=True, exist_ok=True)

default_hea = Path("data/raw/cinc2020/training/georgia/g1/E00001.hea")

if default_hea.exists():
    x, fs, meta = load_wfdb_hea_mat(default_hea)
    source = "wfdb_default_georgia_E00001"
else:
    x, fs, meta = synthetic_ecg(fs=500, seconds=10, pattern="balanced")
    source = "synthetic_balanced_fallback"

result = run_v304_real_inference(
    x_raw=x,
    fs=fs,
    model_path="artifacts/models/inceptiontime_v21_safety.pt",
    threshold_path="artifacts/deep_safety_v21/threshold_profiles_deep.json",
    profile="screening",
    device="cpu",
    source_meta=meta,
)

result["smoke_test"] = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": source,
}

out_path = OUT / "real_inference_smoke_test.json"
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

print("DONE: v3.0.4 real inference bridge smoke test")
print("source:", source)
print("inference_mode:", result.get("inference_mode"))
print("model_loaded:", result.get("model_meta", {}).get("loaded"))
print("positive_labels:", result.get("positive_labels"))
print("recommendation:", result.get("recommendation"))
print("saved:", out_path)
'''

Path("scripts/smoke_test_v304_real_inference_bridge.py").write_text(script, encoding="utf-8")
print("DONE: scripts/smoke_test_v304_real_inference_bridge.py")
