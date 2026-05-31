from pathlib import Path

p = Path("apps/streamlit_cardiotwin_unified_v302.py")
text = p.read_text(encoding="utf-8")

start = text.find("def load_thresholds():")
end = text.find("\ndef unified_ai_panel", start)

if start == -1:
    raise RuntimeError("Could not find def load_thresholds()")
if end == -1:
    raise RuntimeError("Could not find def unified_ai_panel after load_thresholds")

new_func = r'''def _extract_threshold_number(value, default):
    """Extract numeric threshold from flexible JSON structures."""
    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return float(default)

    if isinstance(value, dict):
        # Common possible keys across threshold/calibration exports.
        for key in [
            "threshold",
            "value",
            "tuned_threshold",
            "selected_threshold",
            "operating_threshold",
            "cutoff",
            "cut_off",
            "decision_threshold",
        ]:
            if key in value:
                return _extract_threshold_number(value.get(key), default)

        # Sometimes profile stores a nested class entry.
        for key in ["screening", "balanced", "safety", "default"]:
            if key in value:
                return _extract_threshold_number(value.get(key), default)

        return float(default)

    return float(default)


def _threshold_dict_from_obj(obj, defaults):
    """Try to convert a candidate object into {class: threshold}."""
    if not isinstance(obj, dict):
        return None

    # Direct format:
    # {"NORM": 0.5, "MI": {"threshold": 0.3}, ...}
    if any(k in obj for k in defaults):
        return {
            k: _extract_threshold_number(obj.get(k, defaults[k]), defaults[k])
            for k in defaults
        }

    # Nested formats:
    # {"thresholds": {"NORM": ...}}
    # {"class_thresholds": {"NORM": ...}}
    # {"per_class": {"NORM": ...}}
    for key in [
        "thresholds",
        "class_thresholds",
        "per_class_thresholds",
        "per_class",
        "labels",
        "classes",
    ]:
        if key in obj and isinstance(obj[key], dict):
            found = _threshold_dict_from_obj(obj[key], defaults)
            if found is not None:
                return found

    return None


def load_thresholds():
    defaults = {
        "NORM": 0.50,
        "MI": 0.30,
        "STTC": 0.30,
        "CD": 0.25,
        "HYP": 0.30,
    }

    raw = read_json(V27_THRESHOLDS)

    if not raw:
        return defaults, "fallback_demo_thresholds"

    # 1) Try whole file directly.
    found = _threshold_dict_from_obj(raw, defaults)
    if found is not None:
        return found, "v2.7_thresholds:root"

    # 2) Try common top-level containers.
    for key in [
        "screening",
        "balanced",
        "safety",
        "default",
        "profiles",
        "threshold_profiles",
        "operating_profiles",
        "thresholds",
        "deep_thresholds",
    ]:
        if key not in raw:
            continue

        obj = raw[key]

        # If profiles contains screening profile, prefer it.
        if isinstance(obj, dict) and "screening" in obj:
            found = _threshold_dict_from_obj(obj["screening"], defaults)
            if found is not None:
                return found, f"v2.7_thresholds:{key}.screening"

        found = _threshold_dict_from_obj(obj, defaults)
        if found is not None:
            return found, f"v2.7_thresholds:{key}"

    # 3) Last resort: scan one level deeper.
    if isinstance(raw, dict):
        for key, obj in raw.items():
            if isinstance(obj, dict):
                found = _threshold_dict_from_obj(obj, defaults)
                if found is not None:
                    return found, f"v2.7_thresholds:auto_scan:{key}"

    return defaults, "threshold_file_found_but_unparsed_fallback"
'''

text2 = text[:start] + new_func + "\n" + text[end:]

p.write_text(text2, encoding="utf-8")
print("Patched load_thresholds() in apps/streamlit_cardiotwin_unified_v302.py")
