import asyncio
import argparse
from pathlib import Path

from app.db.session import AsyncSessionLocal
from app.storage.local import LocalStorage
from app.core.config.env_settings import get_embedding_settings, get_log_settings
from app.core.config.json_loader import load_experiment_config
from app.core.config.embedding_config import resolve_embedding_config
from app.core.logging import setup_logging
from app.ingestion.ingest_documents import ingest_documents
from app.embeddings.factory import EmbeddingFactory

async def main(args):
    setup_logging(get_log_settings().log_level)

    embedding_settings = get_embedding_settings()

    experiment_config = load_experiment_config(Path(args.experiment_config))
    embedding_service = EmbeddingFactory.create(
        resolve_embedding_config(
            config=experiment_config.embedding_model,
            settings=embedding_settings
        )   
    )

    async with AsyncSessionLocal() as session:
        await ingest_documents(
            session=session,
            embedding_service=embedding_service,
            ingestion_config=experiment_config.ingestion,
            storage=LocalStorage(experiment_config.ingestion.source_path),
            )
        await session.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", default="data/configs/responses/v1.json")
    args = parser.parse_args()
    asyncio.run(main(args))
