from pathlib import Path
import importlib.util
import inspect
import json

p = Path("src/cardiotwin/runtime/v304_real_inference_bridge.py")

if not p.exists():
    raise FileNotFoundError(p)

spec = importlib.util.spec_from_file_location("v304_real_inference_bridge", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

funcs = []
for name, obj in vars(mod).items():
    if callable(obj) and not name.startswith("_"):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "signature_unavailable"
        funcs.append({"name": name, "signature": sig})

out = {
    "bridge_path": str(p),
    "callable_count": len(funcs),
    "functions": funcs
}

out_path = Path("artifacts/public_multicenter_validation_v33/v331_bridge_api_inventory.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(out, indent=2, ensure_ascii=False))
print("WROTE:", out_path)
