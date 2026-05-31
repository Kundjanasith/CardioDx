from pathlib import Path
import re

path = Path("scripts/evaluate_cinc2020_georgia_external.py")
text = path.read_text(encoding="utf-8")

new_func = r'''def parse_dx(header_path: Path) -> list[str]:
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

        codes = re.findall(r"\d{6,}", value)
        if codes:
            return [c.strip() for c in codes if c.strip()]

        return [x.strip() for x in value.split(",") if x.strip()]

    return []
'''

pattern = r"def parse_dx\(header_path: Path\) -> list\[str\]:.*?\n\n\ndef parse_fs"
replacement = new_func + "\n\n\ndef parse_fs"

text2, n = re.subn(pattern, replacement, text, flags=re.DOTALL)

if n != 1:
    raise RuntimeError(f"parse_dx replacement failed. replacements={n}")

path.write_text(text2, encoding="utf-8")
print("Patched parse_dx successfully.")
