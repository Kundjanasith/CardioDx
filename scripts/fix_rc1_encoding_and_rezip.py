from pathlib import Path
import json
import zipfile

release = Path("artifacts/release_rc1")

# Fix mojibake title in markdown
summary = release / "CARDIOTWIN_RC1_RESEARCH_SUMMARY.md"
text = summary.read_text(encoding="utf-8")
text = text.replace("โ€”", "-")
summary.write_text(text, encoding="utf-8")

# Fix manifest title
manifest = release / "release_manifest.json"
data = json.loads(manifest.read_text(encoding="utf-8"))
if "title" in data:
    data["title"] = data["title"].replace("โ€”", "-")
manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# Rebuild ZIP
zip_path = release / "release_rc1.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(release.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == zip_path.resolve():
            continue
        if p.name.startswith("~$"):
            continue
        z.write(p, (Path("release_rc1") / p.relative_to(release)).as_posix())

print("Fixed title encoding and rebuilt ZIP")
print("ZIP:", zip_path)
print("ZIP size MB:", f"{zip_path.stat().st_size / 1024 / 1024:.2f}")
