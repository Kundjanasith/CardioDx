from pathlib import Path

src = Path("apps/streamlit_dashboard_v25_deep_safety_region.py")
dst = Path("apps/streamlit_dashboard_v251_deep_safety_region.py")

text = src.read_text(encoding="utf-8")

# 1) Add processed sampling rate constant display-friendly text if not already present
text = text.replace(
    'm4.metric("Sampling rate", f"{result[\'fs\']} Hz")',
    'm4.metric("Sampling rate", f"raw {result[\'fs\']} Hz → AI {FS_TARGET:.0f} Hz")'
)

# 2) Replace positive flags logic to not count NORM as abnormal if abnormal classes are positive
old_block = '''positives = pred_df[pred_df["positive"]]["label"].tolist()

        m1, m2, m3, m4 = st.columns(4)
        pred_df = result["prediction_table"]
        positives = pred_df[pred_df["positive"]]["label"].tolist()

        m1.metric("Positive flags", len(positives))'''

new_block = '''pred_df = result["prediction_table"]
        positives_all = pred_df[pred_df["positive"]]["label"].tolist()
        abnormal_flags = [x for x in positives_all if x != "NORM"]

        # Interpret NORM as a reference/normal probability, not an abnormal disease flag.
        # If an abnormal class is positive, do not report NORM as an abnormal flag.
        display_flags = abnormal_flags if abnormal_flags else positives_all

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Abnormal flags", len(abnormal_flags))'''

if old_block in text:
    text = text.replace(old_block, new_block)
else:
    # safer fallback: simple replacements
    text = text.replace(
        'positives = pred_df[pred_df["positive"]]["label"].tolist()',
        'positives_all = pred_df[pred_df["positive"]]["label"].tolist()\n        abnormal_flags = [x for x in positives_all if x != "NORM"]\n        display_flags = abnormal_flags if abnormal_flags else positives_all'
    )
    text = text.replace(
        'm1.metric("Positive flags", len(positives))',
        'm1.metric("Abnormal flags", len(abnormal_flags))'
    )

# 3) Fix warning/info text for flags
text = text.replace(
    '''if positives:
            st.warning("Positive screening flags: " + ", ".join(positives))
        else:
            st.info("No class crossed the selected safety threshold.")''',
    '''if abnormal_flags:
            st.warning("Abnormal screening flags: " + ", ".join(abnormal_flags))
        elif "NORM" in positives_all:
            st.success("Only NORM crossed threshold. No abnormal class crossed the selected safety threshold.")
        else:
            st.info("No class crossed the selected safety threshold.")'''
)

# 4) Replace waveform line_chart with time-axis dataframe
text = text.replace(
    '''wave_df = pd.DataFrame(result["signal"], columns=LEADS)
        st.line_chart(wave_df)''',
    '''wave_df = pd.DataFrame(result["signal"], columns=LEADS)
        wave_df.insert(0, "time_sec", np.arange(len(wave_df)) / FS_TARGET)
        st.line_chart(wave_df, x="time_sec", y=LEADS)'''
)

# 5) Add clinical boundary under 3D title
text = text.replace(
    'st.subheader("3D/4D CardioTwin Heart Region Map")',
    'st.subheader("3D/4D CardioTwin Heart Region Map")\n        st.caption("Pseudo-3D/4D lead-region visual explanation only. This is not patient-specific ECGI or final diagnosis.")'
)

# 6) Add export JSON report button after safety metadata
text = text.replace(
    '''st.json({
            "model_path": str(MODEL_PATH),
            "recommended_default_profile": safety.get("recommended_default_profile"),
            "calibration_note": safety.get("calibration_note"),
            "dx_codes": result["dx_codes"],
            "4d_phase_view": phase,
            "boundary": "Pseudo-3D/4D research visualization, not patient-specific ECGI anatomy."
        })''',
    '''report_payload = {
            "model_path": str(MODEL_PATH),
            "record_id": result["record_id"],
            "model_name": result["model_name"],
            "safety_profile": profile,
            "prediction_table": result["prediction_table"].to_dict(orient="records"),
            "region_decisions": result["region_decisions"],
            "recommended_default_profile": safety.get("recommended_default_profile"),
            "calibration_note": safety.get("calibration_note"),
            "dx_codes": result["dx_codes"],
            "4d_phase_view": phase,
            "boundary": "Pseudo-3D/4D research visualization, not patient-specific ECGI anatomy. Research-use only; not final diagnosis."
        }
        st.json(report_payload)

        st.download_button(
            "Download CardioTwin JSON report",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name=f"cardiotwin_v251_{result['record_id']}_{profile}.json",
            mime="application/json"
        )'''
)

dst.write_text(text, encoding="utf-8")
print("Created:", dst)
