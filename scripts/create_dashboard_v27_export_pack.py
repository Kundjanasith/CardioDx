from pathlib import Path

src = Path("apps/streamlit_dashboard_v251_deep_safety_region.py")
dst = Path("apps/streamlit_dashboard_v27_export_pack.py")

text = src.read_text(encoding="utf-8")

# Make title v2.7
text = text.replace("CardioTwin-AI 12L v2.5", "CardioTwin-AI 12L v2.7")
text = text.replace(
    "Deep Safety + Region Mapper v2.3 + 3D/4D Heart Map",
    "Deep Safety + Region Mapper v2.3 + 3D/4D Heart Map + Export Pack"
)

# Ensure HTML escape import
if "import html" not in text:
    text = text.replace("import json\n", "import json\nimport html\n")

# Add export helper functions before st.set_page_config
helper = r'''

def build_case_report_html(report_payload, prediction_table_html, region_table_html, fig_html):
    record_id = html.escape(str(report_payload.get("record_id", "unknown")))
    profile = html.escape(str(report_payload.get("safety_profile", "unknown")))
    model_name = html.escape(str(report_payload.get("model_name", "unknown")))
    boundary = html.escape(str(report_payload.get("boundary", "")))

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CardioTwin-AI Case Report - {record_id}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
h1, h2 {{ color: #1f2937; }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #eef2ff; margin-right: 8px; }}
.warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.small {{ color: #6b7280; font-size: 0.92em; }}
</style>
</head>
<body>
<h1>CardioTwin-AI 12L v2.7 Case Report</h1>
<p>
<span class="badge">Record: {record_id}</span>
<span class="badge">Model: {model_name}</span>
<span class="badge">Profile: {profile}</span>
</p>

<div class="warning">
<strong>Research-use boundary:</strong> {boundary}
</div>

<h2>Safety-calibrated Prediction</h2>
{prediction_table_html}

<h2>Region Mapper v2.3 Decisions</h2>
{region_table_html}

<h2>3D/4D CardioTwin Snapshot</h2>
<p class="small">Pseudo-3D/4D lead-region visual explanation only. Not patient-specific ECGI.</p>
{fig_html}

<h2>Machine-readable JSON Payload</h2>
<pre>{html.escape(json.dumps(report_payload, indent=2, ensure_ascii=False))}</pre>
</body>
</html>"""


def fig_to_html_safe(fig):
    if fig is None:
        return "<p>3D figure unavailable.</p>"
    try:
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as e:
        return f"<p>Could not export 3D figure: {html.escape(str(e))}</p>"


def fig_to_png_bytes_safe(fig):
    if fig is None:
        return None
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None
'''

if "def build_case_report_html(" not in text:
    text = text.replace("st.set_page_config(", helper + "\n\nst.set_page_config(")

# Replace fig block to retain fig variable and add exports after metadata/report_payload
# v2.5.1 already creates report_payload + download JSON. We add HTML report and 3D HTML/PNG buttons after JSON download_button.
old = '''st.download_button(
            "Download CardioTwin JSON report",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name=f"cardiotwin_v251_{result['record_id']}_{profile}.json",
            mime="application/json"
        )'''

new = '''st.download_button(
            "Download CardioTwin JSON report",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name=f"cardiotwin_v27_{result['record_id']}_{profile}.json",
            mime="application/json"
        )

        # Export pack: HTML report + interactive 3D snapshot + optional PNG
        pred_html = result["prediction_table"].to_html(index=False)
        if result["region_decisions"]:
            region_export_df = pd.DataFrame([
                {
                    "class": d.get("predicted_class"),
                    "class_probability": d.get("class_probability"),
                    "region": d.get("region"),
                    "reason": d.get("reason"),
                    "confidence": d.get("confidence"),
                    "top_region": d.get("top_region", ""),
                    "second_region": d.get("second_region", ""),
                    "margin": d.get("margin", ""),
                }
                for d in result["region_decisions"]
            ])
            region_html = region_export_df.to_html(index=False)
        else:
            region_html = "<p>No abnormal positive class selected for region mapping.</p>"

        fig_html = fig_to_html_safe(fig)
        html_report = build_case_report_html(
            report_payload=report_payload,
            prediction_table_html=pred_html,
            region_table_html=region_html,
            fig_html=fig_html,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.download_button(
                "Download HTML case report",
                data=html_report.encode("utf-8"),
                file_name=f"cardiotwin_v27_case_report_{result['record_id']}_{profile}.html",
                mime="text/html"
            )

        with c2:
            st.download_button(
                "Download interactive 3D HTML",
                data=fig_html.encode("utf-8"),
                file_name=f"cardiotwin_v27_3d_snapshot_{result['record_id']}_{profile}.html",
                mime="text/html"
            )

        with c3:
            png_bytes = fig_to_png_bytes_safe(fig)
            if png_bytes:
                st.download_button(
                    "Download 3D PNG snapshot",
                    data=png_bytes,
                    file_name=f"cardiotwin_v27_3d_snapshot_{result['record_id']}_{profile}.png",
                    mime="image/png"
                )
            else:
                st.caption("PNG snapshot requires kaleido. HTML snapshot is available.")'''

if old in text:
    text = text.replace(old, new)
else:
    print("[WARN] JSON download block not found. Dashboard copied without export extension block.")

dst.write_text(text, encoding="utf-8")
print("Created:", dst)
