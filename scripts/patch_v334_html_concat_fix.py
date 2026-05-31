from pathlib import Path

p = Path("scripts/run_public_failure_case_review_v334.py")
text = p.read_text(encoding="utf-8")

old = '''        "<h2>Top Review Cases</h2>"
        + (top_df.head(100).to_html(index=False) if len(top_df) else "<p>No failure cases.</p>")
        "</body></html>",'''

new = '''        "<h2>Top Review Cases</h2>"
        + (top_df.head(100).to_html(index=False) if len(top_df) else "<p>No failure cases.</p>")
        + "</body></html>",'''

if old not in text:
    raise RuntimeError("Target HTML concat block not found. Please check script content.")

backup = p.with_suffix(".py.v334_html_concat_backup")
backup.write_text(text, encoding="utf-8")

text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

print("DONE: patched missing + before closing HTML string")
print("backup:", backup)
print("script:", p)
