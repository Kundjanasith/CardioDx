from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"
PAGES_DIR = ROOT / "pages"

FILES_TO_COPY = [
    (
        ROOT / "artifacts/public_multicenter_validation_v33/public_multicenter_validation_final_report_v335.html",
        SITE_DIR / "reports/public_multicenter_validation_final_report_v335.html",
    ),
    (
        ROOT / "artifacts/public_multicenter_validation_v33/PUBLIC_MULTICENTER_VALIDATION_FINAL_REPORT_v335.md",
        SITE_DIR / "reports/PUBLIC_MULTICENTER_VALIDATION_FINAL_REPORT_v335.md",
    ),
    (
        ROOT / "artifacts/public_multicenter_validation_v33/PUBLIC_CLAIM_BOUNDARY_AND_NEXT_STEPS_v335.md",
        SITE_DIR / "reports/PUBLIC_CLAIM_BOUNDARY_AND_NEXT_STEPS_v335.md",
    ),
    (
        ROOT / "artifacts/public_multicenter_validation_v33/PUBLIC_PAPER_READY_RESULTS_TABLE_v335.csv",
        SITE_DIR / "reports/PUBLIC_PAPER_READY_RESULTS_TABLE_v335.csv",
    ),
    (
        ROOT / "artifacts/public_multicenter_validation_v33/public_multicenter_validation_final_summary_v335.json",
        SITE_DIR / "reports/public_multicenter_validation_final_summary_v335.json",
    ),
    (
        ROOT / "artifacts/public_multicenter_validation_v33/doctor_in_the_loop_review_template_v334.csv",
        SITE_DIR / "reports/doctor_in_the_loop_review_template_v334.csv",
    ),
    (
        ROOT / "artifacts/release_rc1/cardiotwin_v3_3_5_final_public_multicenter_validation_pack.zip",
        SITE_DIR / "release/cardiotwin_v3_3_5_final_public_multicenter_validation_pack.zip",
    ),
    (
        ROOT / "artifacts/release_rc1/cardiotwin_v3_3_5_final_public_multicenter_validation_manifest.json",
        SITE_DIR / "release/cardiotwin_v3_3_5_final_public_multicenter_validation_manifest.json",
    ),
]


def copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing required file for Pages build: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    copy_file(PAGES_DIR / "index.html", SITE_DIR / "index.html")
    copy_file(PAGES_DIR / ".nojekyll", SITE_DIR / ".nojekyll")

    for source, destination in FILES_TO_COPY:
        copy_file(source, destination)

    print(f"Built GitHub Pages site at {SITE_DIR}")


if __name__ == "__main__":
    main()
