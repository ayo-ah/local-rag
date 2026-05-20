import asyncio
import argparse
from pathlib import Path

from app.core.config.env_settings import (
    get_embedding_settings,
    get_llm_settings,
    get_eval_settings,
    get_log_settings,
)
from app.evaluation.pipeline.evaluation_runner import EvaluationRunner
from app.core.services import AppServices
from app.core.config.embedding_config import resolve_embedding_config
from app.llm.http import LlamaCppHttpLLMService
from app.storage.local import LocalStorage
from app.core.config.json_loader import load_evaluation_config, load_experiment_config
from app.ingestion.ingest_documents import ingest_documents
from app.evaluation.pipeline.inference_runner import run_inference
from app.evaluation.evaluators.llm_evaluator import LLMEvaluator
from app.prompting.builders.evaluation import EvaluationPromptBuilder
from app.core.logging import setup_logging 
from app.query.agents.factory import build_agent
from app.utils.hashing import hash_config
from app.db.session import AsyncSessionLocal

async def run_experiment(args):
    setup_logging(get_log_settings().log_level)

    llm_settings = get_llm_settings()
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
        max_retries=3,
    )
    eval_runner = EvaluationRunner(
        llm_evaluator=llm_evaluator,
    )

    app_services = AppServices(
        embedding_config=resolve_embedding_config(
            config=experiment_config.embedding_model,
            settings=get_embedding_settings()),
        llm_settings=llm_settings,
        llm_config=experiment_config.llm,
        )
    
    
    exp_config_hash = hash_config(experiment_config)
    responses_path = Path(experiment_config.responses_dir) / f"{experiment_config.experiment_name}_{exp_config_hash}.json"

    eval_config.responses_path = responses_path

    eval_config_hash = hash_config(eval_config)
    eval_path = Path(eval_config.evaluation_dir) / f"{eval_config.experiment_name}_{eval_config_hash}.json"

    async with AsyncSessionLocal() as session:
        await ingest_documents(
            session,
            ingestion_config=experiment_config.ingestion,
            embedding_service=app_services.embedding_service,
            storage=LocalStorage(experiment_config.ingestion.source_path),
        )
        await session.commit()

        if not responses_path.exists():
            agent = await build_agent(
                session=session,
                app_services=app_services,
                retrieval_config=experiment_config.retrieval,
                reranker_config=experiment_config.reranker,
                prompt_config=experiment_config.prompt
            )
            await run_inference(
                agent=agent,
                experiment_config=experiment_config,
                )
            
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
