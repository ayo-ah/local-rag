import asyncio
import argparse
from pathlib import Path

from app.evaluation.dataset.generator import SyntheticDatasetGenerator
from app.db.repositories.chunks import ChunkRepository
from app.llm.http import LlamaCppHttpLLMService
from app.db.session import AsyncSessionLocal
from app.core.config.env_settings import get_llm_settings, get_log_settings
from app.prompting.builders.dataset_generation import DatasetPromptBuilder
from app.core.logging import setup_logging, get_logger 
from app.utils.json_io import save_json
from app.core.config.json_loader import load_dataset_generator_config

async def main(args) -> None:
    setup_logging(get_log_settings().log_level)
    logger = get_logger(__name__)

    llm_settings = get_llm_settings()
    config = load_dataset_generator_config(Path(args.generator_config))

    config.output_dir.mkdir(parents=True, exist_ok=True)

    async with AsyncSessionLocal() as session:
        chunks_repo = ChunkRepository(session)

        llm = LlamaCppHttpLLMService(
            base_url=llm_settings.llm_url,
            max_tokens=config.llm.llm_max_tokens,
            temperature=config.llm.llm_temperature,
            stop=config.llm.llm_stop,
	    timeout = 400,
        )

        generator = SyntheticDatasetGenerator(
            chunks_repo=chunks_repo,
            llm=llm,
            questions_per_source=config.questions_per_source,
            prompt_builder= DatasetPromptBuilder(
                config.prompt.root,
                config.prompt.task,
                config.prompt.version,
                )
        )

        all_samples = []

        for source_id in config.source_id:
            sample = await generator.generate_for_source(
                source_id=source_id,
                limit_chunks=config.limit_chunks,
            )
            all_samples.extend(sample)


    serialized = [sample.to_dict() for sample in all_samples]
    save_json(config.output_dir / config.dataset_name, serialized)


    logger.info(f"Dataset_saved_to_{config.output_dir / config.dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_config", default="data/configs/dataset_generation/v2.json")
    args = parser.parse_args()
    asyncio.run(main(args))
