from ..domain.dto import (
    EvalSample,
    RAGResponse,
    GenerationMetrics,
)
from ..evaluators.llm_evaluator import LLMEvaluator


async def evaluate_generation(
    samples: list[EvalSample],
    responses: list[RAGResponse],
    llm_evaluator: LLMEvaluator,
) -> GenerationMetrics:

    ref_map = {s.question_id: s.reference_answer for s in samples}

    scores = []

    for resp in responses:
        reference = ref_map.get(resp.question_id)
        if not reference:
            continue

        score = await llm_evaluator.score(
            question_id=resp.question_id,
            question=resp.question,
            reference=reference,
            answer=resp.answer,
        )
        scores.append(score)

    mean = sum(s.score for s in scores) / len(scores)

    return GenerationMetrics(
        mean_score=mean,
        scores=scores,
    )
