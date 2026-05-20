from pathlib import Path
from app.evaluation.domain.dto import EvalSample
from app.utils.json_io import load_json

def load_dataset(path: Path) -> list[EvalSample]:

    raw = load_json(path)

    return [EvalSample(**item) for item in raw]

