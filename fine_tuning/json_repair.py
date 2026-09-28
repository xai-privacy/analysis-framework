import json
from pathlib import Path


def sanitize_and_fix_jsonl(file_path):
    if not file_path.exists():
        return

    print(f"Repairing {file_path.name}...")

    # Read full file content
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    items = []

    # Case 1: Entire file is a valid JSON array or single object
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = [parsed]
    except Exception:
        # Case 2: File was meant to be JSONL, parse line by line
        lines = content.splitlines()
        current_buffer = ""
        for line in lines:
            current_buffer += line.strip()
            try:
                # Try parsing buffered line
                obj = json.loads(current_buffer)
                items.append(obj)
                current_buffer = ""
            except Exception:
                # Keep accumulating until a valid JSON object completes
                continue

    # Rewrite file with strict 1-line per object JSONL formatting
    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f" successfully saved {len(items)} clean JSONL entries to {file_path.name}"
    )


# Repair data directory
data_dir = Path("./data")
for filename in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
    sanitize_and_fix_jsonl(data_dir / filename)