from __future__ import annotations
from pathlib import Path
import json
import html


def generate_html_report(state: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probs = state.get("class_probabilities", {})
    ranked = state.get("regions", {}).get("ranked_regions", [])
    lead_imp = state.get("lead_importance", {})
    warnings = state.get("sqi", {}).get("warnings", [])
    rows_prob = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.3f}</td></tr>" for k, v in probs.items())
    rows_region = "".join(f"<tr><td>{html.escape(r['region'])}</td><td>{r['risk']:.3f}</td></tr>" for r in ranked)
    rows_leads = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.3f}</td></tr>" for k, v in sorted(lead_imp.items(), key=lambda kv: kv[1], reverse=True))
    warn_html = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings) or "<li>None</li>"
    content = f"""
<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>CardioTwin-AI 12L Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
h1, h2 {{ color: #123; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f2f4f8; }}
.warn {{ background: #fff3cd; border: 1px solid #ffeeba; padding: 12px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>CardioTwin-AI 12L Preliminary ECG Report</h1>
<div class='warn'><b>Clinical boundary:</b> {html.escape(state.get('clinical_boundary','Research-use only.'))}</div>
<div class='card'>
<h2>Summary</h2>
<p><b>Record:</b> {html.escape(str(state.get('record_id')))}</p>
<p><b>Top region:</b> {html.escape(str(state.get('summary',{}).get('top_region')))} ({state.get('summary',{}).get('top_region_risk',0):.3f})</p>
<p><b>Risk level:</b> {html.escape(str(state.get('summary',{}).get('risk_level')))}</p>
<p><b>Signal quality:</b> {state.get('sqi',{}).get('overall_sqi',0):.3f}</p>
</div>
<h2>AI Class Probabilities</h2>
<table><tr><th>Class</th><th>Probability</th></tr>{rows_prob}</table>
<h2>Region Risk</h2>
<table><tr><th>Region</th><th>Risk</th></tr>{rows_region}</table>
<h2>Lead Evidence</h2>
<table><tr><th>Lead</th><th>Importance</th></tr>{rows_leads}</table>
<h2>Signal Quality Warnings</h2>
<ul>{warn_html}</ul>
<h2>Raw State JSON</h2>
<pre>{html.escape(json.dumps(state, indent=2, ensure_ascii=False))}</pre>
</body></html>
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def save_json_report(state: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
