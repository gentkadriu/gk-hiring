import json
from pathlib import Path
from datetime import datetime

def save_report(result: dict, out_dir: str = "reports") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(out_dir) / f"analysis-{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(path)
