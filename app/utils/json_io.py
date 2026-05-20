import json
from pathlib import Path
from uuid import UUID
from datetime import datetime
from typing import Any

def default_serializer(obj: Any):
    if isinstance(obj, UUID) or isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, datetime):
        return obj.isoformat()

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    raise TypeError(f"Type not serializable: {type(obj)}")

def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=default_serializer,
        ),
        encoding="utf-8"
    )
    
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def append_jsonl(path: Path, record: Any):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=default_serializer,
            ) + "\n"
        )
