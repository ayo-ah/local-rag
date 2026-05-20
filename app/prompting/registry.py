from pathlib import Path

def load_prompt(prompt_root: Path, task: str, version: str) -> str:
    path = prompt_root / task / f"{version}.txt"

    if not path.exists():
        raise ValueError(f"Prompt not found: {path}")

    return path.read_text(encoding="utf-8")
