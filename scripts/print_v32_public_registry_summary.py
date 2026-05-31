from pathlib import Path
import json

p = Path("artifacts/public_multicenter_validation_v32/public_dataset_registry_v32.json")
obj = json.loads(p.read_text(encoding="utf-8"))

print("mapping_sources_found:")
for x in obj.get("mapping_sources_found", []):
    print(" -", x)

print("\ncode_to_superclass_size:", obj.get("code_to_superclass_size"))

print("\nready_sources:")
for s in obj.get("sources", []):
    if s.get("readiness") == "ready_for_public_locked_validation_candidate":
        print(" -", s["source_id"], s["label_counts_in_scan"])

print("\nneeds_audit_or_missing:")
for s in obj.get("sources", []):
    if s.get("readiness") != "ready_for_public_locked_validation_candidate":
        print(" -", s["source_id"], "=>", s["readiness"])
