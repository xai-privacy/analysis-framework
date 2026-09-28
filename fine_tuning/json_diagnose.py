import json
from pathlib import Path

data_dir = Path("./data")

for file_path in data_dir.glob("*.jsonl"):
    print(f"\nChecking {file_path.name}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except Exception as e:
                print(f"❌ Syntax Error on Line {i} in {file_path.name}: {e}")
                print(f"   Line Content Snippet: {line[:80]}...")
                break
        else:
            print(f"✅ {file_path.name} passed syntax checks!")