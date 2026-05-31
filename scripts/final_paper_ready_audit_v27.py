from pathlib import Path
import json
import zipfile
import pandas as pd

ROOT = Path(".")
RELEASE = ROOT / "artifacts" / "release_rc1"
AUDIT = ROOT / "artifacts" / "audit_v27"
AUDIT.mkdir(parents=True, exist_ok=True)

def read_text(p):
    p = Path(p)
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def read_json(p):
    p = Path(p)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def exists(rel):
    return (RELEASE / rel).exists()

summary_text = read_text(RELEASE / "CARDIOTWIN_RC1_RESEARCH_SUMMARY.md")
manifest = read_json(RELEASE / "release_manifest.json")
results_csv = RELEASE / "CARDIOTWIN_RC1_RESULTS_TABLE.csv"
zip_path = RELEASE / "release_rc1.zip"

checks = []

def add_check(name, status, detail, severity="info"):
    checks.append({
        "name": name,
        "status": bool(status),
        "severity": severity,
        "detail": detail,
    })

# Core files
required_files = [
    "CARDIOTWIN_RC1_RESEARCH_SUMMARY.md",
    "CARDIOTWIN_RC1_RESULTS_TABLE.csv",
    "release_manifest.json",
    "release_rc1.zip",
    "paper_ready_v26/CARDIOTWIN_METHODS_RESULTS_DRAFT_v26.md",
    "paper_ready_v26/table1_model_comparison_with_cpsc_extra_v26.csv",
    "paper_ready_v26/table1_model_comparison_with_cpsc_extra_v26.md",
    "paper_ready_v26/paper_ready_v26_summary.json",
    "cpsc2018_extra_comparison_v25/cpsc2018_extra_comparison_summary_v25.json",
    "cpsc2018_extra_external_inceptiontime_v25/georgia_deep_external_metrics.json",
    "dashboard_v251/streamlit_dashboard_v251_deep_safety_region.py",
    "region_mapping_v23/region_mapper_v23.py",
    "deep_safety_v21/inceptiontime_v21_safety.pt",
]

for f in required_files:
    add_check(
        f"required_file_exists::{f}",
        exists(f),
        f"{f} exists={exists(f)}",
        "critical" if not exists(f) else "info",
    )

# Summary consistency
required_summary_phrases = [
    "CPSC 2018 Extra All-Five-Label External Validation Addendum",
    "NORM: `49`",
    "MI: `376`",
    "STTC: `1914`",
    "CD: `378`",
    "HYP: `182`",
    "0.7287",
    "0.4631",
    "0.4045",
    "3.3568 ms/record",
    "Research-use only",
]

for phrase in required_summary_phrases:
    add_check(
        f"summary_contains::{phrase}",
        phrase in summary_text,
        f"phrase found={phrase in summary_text}",
        "warning" if phrase not in summary_text else "info",
    )

# Manifest consistency
hl = manifest.get("high_level_metrics", {})
add_check(
    "manifest_release_is_v26",
    manifest.get("release") == "v2.6 RC1",
    f"release={manifest.get('release')}",
    "critical" if manifest.get("release") != "v2.6 RC1" else "info",
)
add_check(
    "manifest_has_all_five_external_dataset",
    hl.get("best_all_five_external_dataset") == "CPSC 2018 Extra external v2.5",
    f"best_all_five_external_dataset={hl.get('best_all_five_external_dataset')}",
    "critical",
)
add_check(
    "manifest_best_all_five_external_auroc",
    abs(float(hl.get("best_all_five_external_valid_auroc", -1)) - 0.7287463225556016) < 1e-9,
    f"best_all_five_external_valid_auroc={hl.get('best_all_five_external_valid_auroc')}",
    "warning",
)

# Results table consistency
if results_csv.exists():
    results = pd.read_csv(results_csv)
    sections = set(results.get("section", []))
    add_check(
        "results_has_cpsc2018_extra_section",
        "cpsc2018_extra_external_validation" in sections,
        f"sections include cpsc2018_extra_external_validation={'cpsc2018_extra_external_validation' in sections}",
        "critical",
    )
else:
    add_check("results_csv_readable", False, "results CSV missing", "critical")

# Paper figures
figs = sorted((RELEASE / "paper_ready_v26").glob("*.png"))
add_check(
    "paper_ready_v26_has_figures",
    len(figs) >= 8,
    f"png_figures={len(figs)}",
    "warning" if len(figs) < 8 else "info",
)

# ZIP check
if zip_path.exists():
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
        add_check("release_zip_readable", True, f"zip_entries={len(names)}")
        add_check(
            "release_zip_contains_manifest",
            any(n.endswith("release_manifest.json") for n in names),
            "manifest in zip",
            "critical",
        )
        add_check(
            "release_zip_contains_v26_paper",
            any("paper_ready_v26/" in n for n in names),
            "paper_ready_v26 in zip",
            "critical",
        )
    except Exception as e:
        add_check("release_zip_readable", False, f"{type(e).__name__}: {e}", "critical")
else:
    add_check("release_zip_exists", False, "release_rc1.zip missing", "critical")

# Count status
critical_failures = [c for c in checks if (not c["status"] and c["severity"] == "critical")]
warnings = [c for c in checks if (not c["status"] and c["severity"] == "warning")]

audit = {
    "version": "audit_v27",
    "project": "CardioTwin-AI 12L",
    "audited_release": manifest.get("release"),
    "n_checks": len(checks),
    "n_critical_failures": len(critical_failures),
    "n_warnings": len(warnings),
    "overall_status": "PASS" if len(critical_failures) == 0 else "FAIL",
    "checks": checks,
    "critical_failures": critical_failures,
    "warnings": warnings,
    "recommendation": (
        "Release is paper-ready for mentor/research review."
        if len(critical_failures) == 0
        else "Fix critical failures before using this release as paper-ready evidence."
    ),
}

(AUDIT / "final_paper_ready_audit_v27.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

lines = []
lines.append("# CardioTwin-AI Final Paper-ready Audit v2.7")
lines.append("")
lines.append(f"- Audited release: `{audit['audited_release']}`")
lines.append(f"- Overall status: **{audit['overall_status']}**")
lines.append(f"- Checks: `{audit['n_checks']}`")
lines.append(f"- Critical failures: `{audit['n_critical_failures']}`")
lines.append(f"- Warnings: `{audit['n_warnings']}`")
lines.append("")
lines.append("## Critical Failures")
if critical_failures:
    for c in critical_failures:
        lines.append(f"- **{c['name']}** — {c['detail']}")
else:
    lines.append("- None")
lines.append("")
lines.append("## Warnings")
if warnings:
    for c in warnings:
        lines.append(f"- **{c['name']}** — {c['detail']}")
else:
    lines.append("- None")
lines.append("")
lines.append("## Recommendation")
lines.append(audit["recommendation"])

(AUDIT / "final_paper_ready_audit_v27.md").write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print(json.dumps({
    "overall_status": audit["overall_status"],
    "n_checks": audit["n_checks"],
    "n_critical_failures": audit["n_critical_failures"],
    "n_warnings": audit["n_warnings"],
    "audit_dir": str(AUDIT),
}, indent=2))
