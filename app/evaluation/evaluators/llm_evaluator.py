import json
from app.llm.base import LLMService
from ..domain.dto import GenerationScore
from app.prompting.builders.evaluation import EvaluationPromptBuilder


class LLMEvaluator:

    def __init__(
        self,
        llm: LLMService,
        prompt_builder: EvaluationPromptBuilder,
        max_retries: int,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.max_retries = max_retries

    async def score(
        self,
        question_id: str,
        question: str,
        reference: str,
        answer: str,
    ) -> GenerationScore:

        prompt = self.prompt_builder.build(
            question=question,
            reference=reference,
            answer=answer,
        )

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            raw = await self.llm.generate(prompt)
            text = raw.strip()

            try:
                parsed = json.loads(text)
                return GenerationScore(
                    question_id=question_id,
                    score=float(parsed["score"]),
                    reasoning=parsed["reasoning"],
                )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = e
                print(f"Attempt {attempt} failed: {text}")
                continue

        
        print(f"All attempts failed for question_id={question_id}: {last_error}")

        return GenerationScore(
            question_id=question_id,
            score=0.0,  
            reasoning="Invalid LLM output",
        )