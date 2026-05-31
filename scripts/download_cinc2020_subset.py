from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests


TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def list_links(url: str, retries: int = 5, sleep_sec: float = 3.0) -> list[str]:
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code in TRANSIENT_STATUS:
                raise requests.HTTPError(f"Transient HTTP {r.status_code}", response=r)
            r.raise_for_status()

            html = r.text
            links = re.findall(r'href="([^"]+)"', html)

            out = []
            for href in links:
                if href.startswith("?") or href.startswith("#"):
                    continue
                if href in ["../", "/"]:
                    continue
                out.append(urljoin(url, href))
            return out

        except Exception as e:
            last_err = e
            wait = sleep_sec * attempt
            print(f"[RETRY DIR {attempt}/{retries}] {url} | {e} | wait={wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed to list directory after retries: {url}") from last_err


def is_file_url(url: str) -> bool:
    lower = url.lower()
    return lower.endswith((".hea", ".mat", ".dat", ".csv", ".txt", ".md", ".json"))


def download_file(
    url: str,
    out_path: Path,
    chunk_size: int = 1024 * 1024,
    retries: int = 8,
    sleep_sec: float = 3.0,
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        existing = out_path.stat().st_size if out_path.exists() else 0
        headers = {}

        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        try:
            with requests.get(url, headers=headers, stream=True, timeout=180) as r:
                if r.status_code == 416:
                    print(f"[SKIP complete] {out_path}")
                    return True

                if r.status_code in TRANSIENT_STATUS:
                    raise requests.HTTPError(f"Transient HTTP {r.status_code}", response=r)

                if r.status_code == 200 and existing > 0:
                    # Server did not honor resume. Redownload safely.
                    existing = 0

                r.raise_for_status()

                mode = "ab" if existing > 0 and r.status_code == 206 else "wb"

                tmp_path = out_path
                with open(tmp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            print(f"[OK] {out_path}")
            return True

        except Exception as e:
            wait = sleep_sec * attempt
            print(f"[RETRY FILE {attempt}/{retries}] {url} | {e} | wait={wait:.1f}s")
            time.sleep(wait)

    print(f"[FAILED] {url}")
    return False


def crawl_and_download(
    base_url: str,
    out_dir: Path,
    max_files: int | None = None,
    delay_sec: float = 0.05,
):
    seen_dirs = set()
    queue = [base_url if base_url.endswith("/") else base_url + "/"]
    downloaded = 0
    failed: list[str] = []

    while queue:
        url = queue.pop(0)

        if url in seen_dirs:
            continue

        seen_dirs.add(url)
        print(f"\n[DIR] {url}")

        try:
            links = list_links(url)
        except Exception as e:
            print(f"[FAILED DIR] {url} | {e}")
            failed.append(url)
            continue

        for link in links:
            parsed = urlparse(link)
            name = unquote(Path(parsed.path).name)

            if not name:
                continue

            if link.endswith("/"):
                if "/files/challenge-2020/1.0.2/" in link:
                    queue.append(link)
                continue

            if not is_file_url(link):
                continue

            rel = link.split("/files/challenge-2020/1.0.2/")[-1]
            out_path = out_dir / rel

            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"[SKIP exists] {out_path}")
                continue

            ok = download_file(link, out_path)
            if ok:
                downloaded += 1
            else:
                failed.append(link)

            if delay_sec > 0:
                time.sleep(delay_sec)

            if max_files is not None and downloaded >= max_files:
                print(f"[STOP] reached max_files={max_files}")
                write_failed(out_dir, failed)
                return

    write_failed(out_dir, failed)
    print(f"\n[DONE] downloaded_new_files={downloaded}, failed={len(failed)}")


def write_failed(out_dir: Path, failed: list[str]):
    if not failed:
        return

    path = out_dir / "failed_downloads.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(failed), encoding="utf-8")
    print(f"[WARN] failed URLs saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="training/georgia", help="Example: training/georgia or training/cpsc_2018")
    ap.add_argument("--out-dir", default="data/raw/cinc2020")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--delay-sec", type=float, default=0.05)
    args = ap.parse_args()

    base = f"https://physionet.org/files/challenge-2020/1.0.2/{args.subset.strip('/')}/"
    crawl_and_download(
        base_url=base,
        out_dir=Path(args.out_dir),
        max_files=args.max_files,
        delay_sec=args.delay_sec,
    )


if __name__ == "__main__":
    main()
