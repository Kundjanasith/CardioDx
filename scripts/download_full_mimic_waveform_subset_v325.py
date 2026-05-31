from pathlib import Path
import argparse
import getpass
import json
import time
from urllib.parse import urlparse

import requests

BASE_MARKER = "/files/mimic-iv-ecg/1.0/"

def url_to_rel_path(url):
    parsed = urlparse(url)
    path = parsed.path
    if BASE_MARKER not in path:
        raise ValueError(f"Bad URL: {url}")
    rel = path.split(BASE_MARKER, 1)[1]
    return Path(rel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="artifacts/label_supported_external_validation_v32/full_mimic_waveform_subset_download_urls_v325.txt")
    ap.add_argument("--out-root", default="data/raw/mimic_iv_ecg")
    ap.add_argument("--delay-sec", type=float, default=0.05)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    urls_path = Path(args.urls)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    urls = [x.strip() for x in urls_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    print("[INFO] URLs:", len(urls))
    print("[INFO] Output root:", out_root)

    username = input("PhysioNet username: ").strip()
    password = getpass.getpass("PhysioNet password: ")

    session = requests.Session()
    session.auth = (username, password)

    ok = 0
    skipped = 0
    failed = 0
    results = []

    for i, url in enumerate(urls, start=1):
        rel = url_to_rel_path(url)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{i}/{len(urls)}] SKIP {out_path}")
            skipped += 1
            results.append({"url": url, "path": str(out_path), "status": "skipped_exists"})
            continue

        print(f"[{i}/{len(urls)}] GET {url}")

        try:
            with session.get(url, stream=True, timeout=args.timeout) as r:
                if r.status_code != 200:
                    print("[ERROR] HTTP", r.status_code)
                    failed += 1
                    results.append({"url": url, "path": str(out_path), "status": "error", "http_status": r.status_code})
                    continue

                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                tmp.replace(out_path)
                print("[OK]", out_path, out_path.stat().st_size, "bytes")
                ok += 1
                results.append({"url": url, "path": str(out_path), "status": "ok", "size_bytes": out_path.stat().st_size})

        except Exception as e:
            print("[ERROR]", repr(e))
            failed += 1
            results.append({"url": url, "path": str(out_path), "status": "error", "error": repr(e)})

        time.sleep(args.delay_sec)

    report = {
        "urls_total": len(urls),
        "downloaded_ok": ok,
        "skipped_exists": skipped,
        "failed": failed,
        "out_root": str(out_root),
        "results": results,
    }

    report_path = Path("artifacts/label_supported_external_validation_v32/full_mimic_waveform_subset_download_report_v325.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DONE")
    print("downloaded_ok:", ok)
    print("skipped_exists:", skipped)
    print("failed:", failed)
    print("report:", report_path)

if __name__ == "__main__":
    main()
