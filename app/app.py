from fastapi import FastAPI

from app.api.router import api_router
from app.core.lifespan import lifespan
from app.core.logging import setup_logging
from app.core.config.env_settings import get_log_settings


def create_app() -> FastAPI:
    setup_logging(level=get_log_settings().log_level)

    app = FastAPI(
        title="RAG Service",
        lifespan=lifespan,
    )

    app.include_router(api_router)

    return app
