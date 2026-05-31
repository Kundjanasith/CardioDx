from __future__ import annotations
import numpy as np
import pandas as pd
from cardiotwin.constants import LEAD_TO_REGION


def infer_reference_regions(labels: dict) -> list[str]:
    refs = []
    if labels.get("MI", 0) or labels.get("STTC", 0):
        # PTB-XL superclasses do not provide exact location; these are broad eligible regions.
        refs += ["inferior", "anterior", "septal", "lateral"]
    if labels.get("CD", 0):
        refs += ["global_conduction"]
    if labels.get("NORM", 0) and not refs:
        refs += ["uncertain"]
    return sorted(set(refs)) or ["uncertain"]


def region_mapping_metrics(cases: list[dict]) -> tuple[dict, pd.DataFrame]:
    rows = []
    top1_hits, top2_hits, agreements = [], [], []
    confusion = {}
    for c in cases:
        ref_regions = c.get("reference_regions") or infer_reference_regions(c.get("labels", {}))
        ranked = [r["region"] if isinstance(r, dict) else r[0] for r in c.get("ranked_regions", [])]
        pred1 = ranked[0] if ranked else "uncertain"
        pred2 = ranked[:2]
        top1 = int(pred1 in ref_regions)
        top2 = int(any(r in ref_regions for r in pred2))
        top1_hits.append(top1)
        top2_hits.append(top2)
        # lead-region agreement: top lead's mapped region appears in top predicted regions
        lead_imp = c.get("lead_importance", {})
        top_lead = max(lead_imp, key=lead_imp.get) if lead_imp else None
        top_lead_region = LEAD_TO_REGION.get(top_lead, "uncertain")
        agree = int(top_lead_region in pred2)
        agreements.append(agree)
        key = (ref_regions[0], pred1)
        confusion[key] = confusion.get(key, 0) + 1
        rows.append({
            "record_id": c.get("record_id"),
            "reference_regions": ";".join(ref_regions),
            "pred_top1": pred1,
            "pred_top2": ";".join(pred2),
            "top1_hit": top1,
            "top2_hit": top2,
            "top_lead": top_lead,
            "top_lead_region": top_lead_region,
            "lead_region_agreement": agree,
        })
    overall = {
        "region_top1_hit": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "region_top2_hit": float(np.mean(top2_hits)) if top2_hits else float("nan"),
        "lead_region_agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "region_confusion_matrix": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
    }
    return overall, pd.DataFrame(rows)
