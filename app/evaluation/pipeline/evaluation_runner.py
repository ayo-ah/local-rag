
from pathlib import Path

from ..domain.dto import EvalBundle
from ..evaluators.llm_evaluator import LLMEvaluator
from ..metrics.retrieval import evaluate_retrieval
from ..metrics.generation import evaluate_generation
from app.evaluation.artifacts import write_evaluation_artifact, read_inference_artifact
from app.evaluation.dataset.loader import load_dataset
from app.core.config.experiment_config import EvaluationConfig

class EvaluationRunner:

    def __init__(self, llm_evaluator: LLMEvaluator):
        self.llm_evaluator = llm_evaluator

    async def run(
        self,
        eval_config: EvaluationConfig,
    ):

        samples = load_dataset(Path(eval_config.dataset_path))
        responses = read_inference_artifact(Path(eval_config.responses_path))
        retrieval_metrics = evaluate_retrieval(samples, responses)


        generation_metrics = await evaluate_generation(
            samples,
            responses,
            self.llm_evaluator,
        )

        bundle = EvalBundle(
            config=eval_config,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            responses=responses,
        )

        write_evaluation_artifact(bundle)
