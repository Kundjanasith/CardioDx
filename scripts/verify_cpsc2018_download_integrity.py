from pathlib import Path
import argparse
import pandas as pd
from scipy.io import loadmat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/cinc2020/training/cpsc_2018")
    ap.add_argument("--delete-bad", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path("artifacts/external_validation/cpsc2018_download_qc")
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = []

    heas = sorted(root.rglob("*.hea"))
    mats = sorted(root.rglob("*.mat"))

    hea_stems = {p.stem for p in heas}
    mat_stems = {p.stem for p in mats}

    for hea in heas:
        mat = hea.with_suffix(".mat")
        if not mat.exists():
            problems.append({
                "record": hea.stem,
                "path": str(mat),
                "problem": "missing_mat_for_header"
            })

    for mat in mats:
        hea = mat.with_suffix(".hea")
        if not hea.exists():
            problems.append({
                "record": mat.stem,
                "path": str(mat),
                "problem": "mat_without_header"
            })
            if args.delete_bad:
                mat.unlink(missing_ok=True)
            continue

        try:
            data = loadmat(mat)
            if "val" not in data:
                problems.append({
                    "record": mat.stem,
                    "path": str(mat),
                    "problem": "mat_missing_val_key"
                })
                if args.delete_bad:
                    mat.unlink(missing_ok=True)
                continue

            x = data["val"]
            if len(x.shape) != 2 or 12 not in x.shape:
                problems.append({
                    "record": mat.stem,
                    "path": str(mat),
                    "problem": f"unexpected_mat_shape_{x.shape}"
                })
                if args.delete_bad:
                    mat.unlink(missing_ok=True)

        except Exception as e:
            problems.append({
                "record": mat.stem,
                "path": str(mat),
                "problem": f"corrupt_mat:{type(e).__name__}:{e}"
            })
            if args.delete_bad:
                mat.unlink(missing_ok=True)

    missing_hea = sorted(mat_stems - hea_stems)
    missing_mat = sorted(hea_stems - mat_stems)

    for stem in missing_hea:
        problems.append({
            "record": stem,
            "path": "",
            "problem": "stem_has_mat_but_missing_hea"
        })

    for stem in missing_mat:
        problems.append({
            "record": stem,
            "path": "",
            "problem": "stem_has_hea_but_missing_mat"
        })

    df = pd.DataFrame(problems)
    report = out_dir / "cpsc2018_download_integrity_report.csv"
    df.to_csv(report, index=False)

    print("hea_count:", len(heas))
    print("mat_count:", len(mats))
    print("paired_record_count:", len(hea_stems & mat_stems))
    print("problem_count:", len(df))
    print("report:", report)

    if len(df):
        print(df.head(50).to_string(index=False))
        if args.delete_bad:
            print("Bad/corrupt files deleted. Rerun downloader to repair.")
        else:
            print("Run again with --delete-bad if corrupt partial files are found.")
    else:
        print("QC PASS: CPSC 2018 download looks complete and readable.")

if __name__ == "__main__":
    main()
