from pathlib import Path

p = Path("src/cardiotwin/runtime/v304_real_inference_bridge.py")
text = p.read_text(encoding="utf-8")

# ---------------------------------------------------------------------
# Patch 1: replace make_model_flexible()
# ---------------------------------------------------------------------
start = text.find("def make_model_flexible(")
end = text.find("\ndef load_torch_model", start)

if start == -1 or end == -1:
    raise RuntimeError("Cannot find make_model_flexible block")

new_make_model = r'''def make_model_flexible(make_deep_model, n_classes: int = 5):
    """
    Construct v2.7 deep model flexibly.

    The v2.7 factory requires a required 'name' argument.
    Typical working name is expected to be 'inceptiontime', but we try
    several safe aliases without modifying frozen v2.7 code.
    """
    if make_deep_model is None:
        return None, "make_deep_model_unavailable"

    import inspect

    try:
        sig = str(inspect.signature(make_deep_model))
    except Exception:
        sig = "signature_unavailable"

    candidate_names = [
        "inceptiontime",
        "InceptionTime",
        "inception_time",
        "inceptiontime_v21",
        "deep_inceptiontime",
        "deep_ecg",
        "cnn1d",
    ]

    attempts = []

    for name in candidate_names:
        attempts.extend([
            ((name,), {"n_classes": n_classes}),
            ((name,), {"num_classes": n_classes}),
            ((name,), {"out_dim": n_classes}),
            ((name,), {"in_channels": 12, "n_classes": n_classes}),
            ((name,), {"in_chans": 12, "n_classes": n_classes}),
            ((name,), {}),
            ((), {"name": name, "n_classes": n_classes}),
            ((), {"name": name, "num_classes": n_classes}),
            ((), {"model_name": name, "n_classes": n_classes}),
            ((name, n_classes), {}),
            ((name, 12, n_classes), {}),
        ])

    errors = []
    for args, kwargs in attempts:
        try:
            model = make_deep_model(*args, **kwargs)
            return model, f"make_deep_model(args={args}, kwargs={kwargs}, signature={sig})"
        except Exception as e:
            errors.append(f"args={args}, kwargs={kwargs}: {repr(e)}")

    return None, "make_deep_model_failed_signature=" + sig + " | " + " | ".join(errors[:8])
'''

text = text[:start] + new_make_model + "\n" + text[end:]


# ---------------------------------------------------------------------
# Patch 2: expand checkpoint state-dict key search
# ---------------------------------------------------------------------
old_keys = 'for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:'
new_keys = 'for key in ["state_dict", "model_state_dict", "model_state", "model_weights", "network_state_dict", "module", "model", "net", "weights", "checkpoint"]:'

if old_keys in text:
    text = text.replace(old_keys, new_keys)
else:
    print("WARN: state_dict key list not found; skip key-list patch")


# ---------------------------------------------------------------------
# Patch 3: replace try_region_mapper_v23()
# ---------------------------------------------------------------------
start = text.find("def try_region_mapper_v23(")
end = text.find("\ndef run_v304_real_inference", start)

if start == -1 or end == -1:
    raise RuntimeError("Cannot find try_region_mapper_v23 block")

new_region = r'''def try_region_mapper_v23(map_prediction_to_region, x_ai: np.ndarray, probabilities: dict, positive_labels: list[str]) -> tuple[list[dict], dict]:
    """
    Call Region Mapper v2.3 with flexible signatures.

    Observed v2.3 behavior suggests a likely signature:
        map_prediction_to_region(lead_evidence, predicted_class, class_probability)

    so this patch supplies a lead-evidence dictionary first.
    """
    meta = {"used": False, "errors": []}

    if map_prediction_to_region is None:
        meta["errors"].append("region_mapper_v23_import_unavailable")
        return [], meta

    import inspect

    try:
        meta["signature"] = str(inspect.signature(map_prediction_to_region))
    except Exception:
        meta["signature"] = "signature_unavailable"

    lead_evidence = lead_amplitudes(x_ai)
    region_evidence = fallback_region_mapper(x_ai).get("scores", {})

    decisions = []

    for label in positive_labels:
        prob = float(probabilities.get(label, 0.0))

        call_attempts = [
            # Most likely v2.3 signature
            ("lead_evidence_positional", lambda: map_prediction_to_region(lead_evidence, label, prob)),

            # Keyword variants
            ("lead_evidence_keywords", lambda: map_prediction_to_region(
                lead_evidence=lead_evidence,
                predicted_class=label,
                class_probability=prob,
            )),
            ("evidence_keywords", lambda: map_prediction_to_region(
                evidence=lead_evidence,
                predicted_class=label,
                class_probability=prob,
            )),
            ("region_scores_keywords", lambda: map_prediction_to_region(
                region_scores=region_evidence,
                predicted_class=label,
                class_probability=prob,
            )),

            # Fallback signatures
            ("predicted_keywords", lambda: map_prediction_to_region(
                predicted_class=label,
                class_probability=prob,
            )),
            ("two_positional", lambda: map_prediction_to_region(label, prob)),
        ]

        for attempt_name, fn in call_attempts:
            try:
                res = fn()
                if isinstance(res, dict):
                    d = dict(res)
                else:
                    d = {"raw_result": str(res)}

                d["class"] = label
                d["class_probability"] = prob
                d["region_mapper_attempt"] = attempt_name
                decisions.append(d)
                meta["used"] = True
                break

            except Exception as e:
                meta["errors"].append(f"{label}/{attempt_name}: {repr(e)}")

    return decisions, meta
'''

text = text[:start] + new_region + "\n" + text[end:]

p.write_text(text, encoding="utf-8")

print("DONE: patched v304 bridge -> v3.0.4.1")
print("Patched:", p)
