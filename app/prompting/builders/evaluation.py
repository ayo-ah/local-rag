from pathlib import Path

from app.prompting.builders.base import PromptBuilder
from app.prompting.registry import load_prompt

class EvaluationPromptBuilder(PromptBuilder):

    def __init__(self, prompt_root: Path, task: str, version: str):
        self.template = load_prompt(prompt_root, task, version)

    def build(self, question: str, reference: str, answer: str) -> str:

        return self.template.format(
            question=question,
            reference=reference,
            answer=answer,
        )
