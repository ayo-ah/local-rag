import json

from app.llm.base import LLMService


class MockLLMService(LLMService):

    async def generate(self, prompt: str, **_) -> str:

        payload = {
            "question": "Тестовый фактический вопрос?",
            "answer": "Тестовый ответ на основе текста.",
            "score": 4,
            "reasoning": "Ответ логично следует из текста."
        }

        return json.dumps(payload, ensure_ascii=False)
