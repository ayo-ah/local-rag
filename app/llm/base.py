from typing import Protocol

class LLMService(Protocol):
    max_tokens: str
    temperature: float
    stop: list[str]

    async def generate(
        self,
        prompt: str,):
        ...
