from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.services import AppServices
from app.core.config.embedding_config import resolve_embedding_config
from app.core.config.env_settings import get_llm_settings, get_embedding_settings, get_app_settings
from app.core.config.json_loader import load_experiment_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = get_app_settings()
    cfg = load_experiment_config(config_path)

    app.state.experiment_config = cfg
    app.state.services = AppServices(
        llm_settings=get_llm_settings(),
        llm_config=cfg.llm,
        embedding_config=resolve_embedding_config(
            config=cfg.embedding_model,
            settings=get_embedding_settings())
    )

    yield

