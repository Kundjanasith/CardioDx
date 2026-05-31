from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/"

DEFAULT_CANDIDATES = [
    "training/ptb",
    "training/ptb-xl",
    "training/st_petersburg_incart",
    "training/chapman_shaoxing",
    "training/ningbo",
    "training/cpsc_2018",
    "training/cpsc_2018_extra",
    "training/georgia",
]

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_harmonization_map(path: Path) -> dict[str, str]:
    if not path.exists():
        print(f"[WARN] harmonization map not found: {path}")
        return {}

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # robust column names
    code_col = "code" if "code" in df.columns else df.columns[0]
    class_col = "ptbxl_superclass" if "ptbxl_superclass" in df.columns else None
    decision_col = "decision" if "decision" in df.columns else None

    if class_col is None:
        raise RuntimeError(f"Cannot find ptbxl_superclass column in {path}. Columns={df.columns.tolist()}")

    code_to_class = {}
    for _, r in df.iterrows():
        code = str(r.get(code_col, "")).strip()
        cls = str(r.get(class_col, "")).strip()
        decision = str(r.get(decision_col, "include")).strip().lower() if decision_col else "include"

        if not code or code.lower() == "nan":
            continue
        if decision != "include":
            continue
        if cls not in TARGET_CLASSES:
            continue

        code_to_class[code] = cls

    return code_to_class


def parse_dx_from_header_text(text: str) -> list[str]:
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue

        body = s[1:].strip()
        key, sep, value = body.partition(":")
        if not sep:
            continue

        if key.strip().lower() not in {"dx", "diagnosis"}:
            continue

        return re.findall(r"\d{6,}", value)

    return []


def request_text(url: str, timeout: int = 60, max_retries: int = 5, delay_sec: float = 0.2) -> str | None:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text

            last_err = f"status={r.status_code}"

            # retry common transient statuses
            if r.status_code in {429, 500, 502, 503, 504}:
                time.sleep(delay_sec * attempt)
                continue

            return None

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(delay_sec * attempt)

    print(f"[WARN] failed request after retries: {url} | {last_err}")
    return None


def list_links(url: str, delay_sec: float) -> list[str]:
    html = request_text(url, delay_sec=delay_sec)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a"):
        href = a.get("href", "")
        if not href or href in {"../", "./"}:
            continue
        links.append(urljoin(url, href))

    return links


def collect_hea_urls(subset: str, delay_sec: float) -> list[str]:
    start = urljoin(BASE_URL, subset.strip("/") + "/")
    stack = [start]
    seen_dirs = set()
    hea_urls = []

    while stack:
        url = stack.pop()
        if url in seen_dirs:
            continue
        seen_dirs.add(url)

        links = list_links(url, delay_sec=delay_sec)
        for link in links:
            if link.endswith("/"):
                stack.append(link)
            elif link.endswith(".hea"):
                hea_urls.append(link)

        time.sleep(delay_sec)

    return sorted(set(hea_urls))


def cache_path_for_url(cache_root: Path, subset: str, url: str) -> Path:
    record = Path(url).name
    safe_subset = subset.replace("/", "__")
    return cache_root / safe_subset / record


def load_or_download_header(url: str, subset: str, cache_root: Path, delay_sec: float) -> str | None:
    p = cache_path_for_url(cache_root, subset, url)
    if p.exists() and p.stat().st_size > 0:
        return p.read_text(encoding="utf-8", errors="ignore")

    text = request_text(url, delay_sec=delay_sec)
    if text:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        time.sleep(delay_sec)

    return text


def classify_codes(codes: list[str], code_to_class: dict[str, str]) -> tuple[list[str], list[str]]:
    mapped = sorted(set(code_to_class[c] for c in codes if c in code_to_class))
    unmapped = sorted(set(c for c in codes if c not in code_to_class))
    return mapped, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-path", default="configs/cinc2020_to_ptbxl_superclass_map_v21.csv")
    ap.add_argument("--out-dir", default="artifacts/dataset_probe_mi_hyp_v25")
    ap.add_argument("--delay-sec", type=float, default=0.15)
    ap.add_argument("--max-headers-per-subset", type=int, default=0, help="0 = full subset")
    ap.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = out / "header_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    code_to_class = load_harmonization_map(Path(args.map_path))
    print(f"[INFO] loaded harmonization mappings: {len(code_to_class)} include codes")

    subset_rows = []
    detail_rows = []
    unmapped_rows = []

    for subset in args.candidates:
        print(f"\n[PROBE] {subset}")

        hea_urls = collect_hea_urls(subset, delay_sec=args.delay_sec)
        if args.max_headers_per_subset and args.max_headers_per_subset > 0:
            hea_urls = hea_urls[:args.max_headers_per_subset]

        print(f"[INFO] headers found: {len(hea_urls)}")

        counts = {c: 0 for c in TARGET_CLASSES}
        usable = 0
        failed = 0
        unmapped_counter = {}

        for i, url in enumerate(hea_urls, start=1):
            text = load_or_download_header(url, subset, cache_root, delay_sec=args.delay_sec)
            if not text:
                failed += 1
                continue

            codes = parse_dx_from_header_text(text)
            mapped, unmapped = classify_codes(codes, code_to_class)

            for c in unmapped:
                unmapped_counter[c] = unmapped_counter.get(c, 0) + 1

            if mapped:
                usable += 1
                for lab in mapped:
                    counts[lab] += 1

            detail_rows.append({
                "subset": subset,
                "record_id": Path(url).stem,
                "url": url,
                "dx_codes": ",".join(codes),
                "mapped_labels": ",".join(mapped),
                "unmapped_codes": ",".join(unmapped),
            })

            if i % 500 == 0:
                print(f"  {i}/{len(hea_urls)} usable={usable} MI={counts['MI']} HYP={counts['HYP']}")

        for code, n in sorted(unmapped_counter.items(), key=lambda x: x[1], reverse=True)[:100]:
            unmapped_rows.append({
                "subset": subset,
                "code": code,
                "count": n,
            })

        priority = (
            counts["MI"] * 3.0
            + counts["HYP"] * 3.0
            + counts["STTC"] * 0.25
            + counts["CD"] * 0.15
            + counts["NORM"] * 0.05
        )

        subset_rows.append({
            "subset": subset,
            "hea_headers_seen": len(hea_urls),
            "usable_mapped_records": usable,
            "failed_headers": failed,
            "NORM": counts["NORM"],
            "MI": counts["MI"],
            "STTC": counts["STTC"],
            "CD": counts["CD"],
            "HYP": counts["HYP"],
            "mi_hyp_support": counts["MI"] + counts["HYP"],
            "mi_hyp_priority_score": priority,
            "recommendation": (
                "download_next"
                if (counts["MI"] + counts["HYP"]) >= 20
                else "low_mi_hyp_under_current_mapping"
            ),
        })

    summary = pd.DataFrame(subset_rows).sort_values(
        ["mi_hyp_support", "mi_hyp_priority_score", "usable_mapped_records"],
        ascending=False
    )

    details = pd.DataFrame(detail_rows)
    unmapped_df = pd.DataFrame(unmapped_rows)

    summary_path = out / "candidate_dataset_mi_hyp_support_summary.csv"
    details_path = out / "candidate_dataset_header_probe_details.csv"
    unmapped_path = out / "candidate_dataset_top_unmapped_codes.csv"
    report_path = out / "dataset_probe_mi_hyp_report.json"

    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    unmapped_df.to_csv(unmapped_path, index=False)

    best = summary.head(5).to_dict(orient="records")

    report = {
        "version": "dataset_probe_mi_hyp_v25",
        "goal": "Find external subsets with stronger MI/HYP support before downloading full waveform .mat files.",
        "base_url": BASE_URL,
        "harmonization_map": args.map_path,
        "candidates": args.candidates,
        "best_candidates": best,
        "outputs": [
            str(summary_path),
            str(details_path),
            str(unmapped_path),
            str(report_path),
        ],
        "interpretation": (
            "High MI/HYP support means the subset is useful for the next external validation target. "
            "If all subsets show low MI/HYP under current mapping, inspect candidate_dataset_top_unmapped_codes.csv "
            "and improve harmonization before downloading more waveform data."
        ),
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
