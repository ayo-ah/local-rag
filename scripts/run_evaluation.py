import asyncio
import argparse
from pathlib import Path

from app.core.config.env_settings import (
    get_eval_settings,
    get_log_settings,
)
from app.evaluation.pipeline.evaluation_runner import EvaluationRunner
from app.llm.http import LlamaCppHttpLLMService
from app.core.config.json_loader import load_evaluation_config, load_experiment_config
from app.evaluation.evaluators.llm_evaluator import LLMEvaluator
from app.prompting.builders.evaluation import EvaluationPromptBuilder
from app.core.logging import setup_logging 
from app.utils.hashing import hash_config

async def run_experiment(args):
    setup_logging(get_log_settings().log_level)

    eval_settings = get_eval_settings()

    experiment_config = load_experiment_config(Path(args.experiment_config))
    eval_config = load_evaluation_config(Path(args.evaluation_config))

    judge_llm = LlamaCppHttpLLMService(
        base_url=eval_settings.evaluator_llm_base_url
    )
    
    llm_evaluator = LLMEvaluator(
        llm=judge_llm,
        prompt_builder=EvaluationPromptBuilder(
            eval_config.evaluator_prompt.root,
            eval_config.evaluator_prompt.task, 
            eval_config.evaluator_prompt.version,
            ),
        max_retries=10,
    )
    eval_runner = EvaluationRunner(
        llm_evaluator=llm_evaluator,
    )
    
    
    exp_config_hash = hash_config(experiment_config)
    responses_path = Path(experiment_config.responses_dir) / f"{experiment_config.experiment_name}_{exp_config_hash}.json"

    eval_config.responses_path = responses_path

    eval_config_hash = hash_config(eval_config)
    eval_path = Path(eval_config.evaluation_dir) / f"{eval_config.experiment_name}_{eval_config_hash}.json"

            
    if not eval_path.exists():
        await eval_runner.run(
            eval_config=eval_config,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", default="data/configs/responses/v1.json")
    parser.add_argument("--evaluation_config", default="data/configs/evaluations/v1.json")
    args = parser.parse_args()
    asyncio.run(run_experiment(args))
