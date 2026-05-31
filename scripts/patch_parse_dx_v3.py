from pathlib import Path

path = Path("scripts/evaluate_cinc2020_georgia_external.py")
text = path.read_text(encoding="utf-8")

new_func = '''def parse_dx(header_path: Path) -> list[str]:
    text = header_path.read_text(errors="ignore", encoding="utf-8-sig")

    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue

        # Accept all common variants:
        # #Dx: 426783006
        # # Dx: 426783006
        # #Dx : 426783006
        # # Diagnosis: 426783006
        body = s[1:].strip()
        key, sep, value = body.partition(":")
        if not sep:
            continue

        key = key.strip().lower()
        if key not in {"dx", "diagnosis"}:
            continue

        import re
        codes = re.findall(r"\\d{6,}", value)
        if codes:
            return [c.strip() for c in codes if c.strip()]

        return [x.strip() for x in value.split(",") if x.strip()]

    return []
'''

start = text.find("def parse_dx(")
if start == -1:
    raise RuntimeError("Could not find def parse_dx(...)")

end = text.find("\ndef parse_fs", start)
if end == -1:
    raise RuntimeError("Could not find def parse_fs after parse_dx")

# Keep def parse_fs and everything after it.
text2 = text[:start] + new_func + "\n\n" + text[end + 1:]

path.write_text(text2, encoding="utf-8")
print("Patched parse_dx successfully with v3.")
