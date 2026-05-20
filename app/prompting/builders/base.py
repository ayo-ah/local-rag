from typing import Protocol

class PromptBuilder(Protocol):
    def build(self, **kwargs) -> str:
        ...
