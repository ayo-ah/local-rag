from pathlib import Path
from app.prompting.builders.base import PromptBuilder
from app.prompting.registry import load_prompt
from app.query.domain.dto import QueryContext

class RAGPromptBuilder(PromptBuilder):

    def __init__(self, prompt_root: Path, task: str, version: str):
        self.template = load_prompt(prompt_root, task, version)

    def build(self, context: QueryContext, context_chunks: list[str]) -> str:
        context_block = "\n\n".join(context_chunks)

        return self.template.format(
            query=context.query.text,
            context=context_block
        )
