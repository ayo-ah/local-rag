from dataclasses import dataclass
from typing import Literal
from .experiment_config import EmbeddingConfig
from .env_settings import EmbeddingSettings
@dataclass
class ResolvedEmbeddingConfig:
    provider: Literal["local", "http"]
    model_name: str
    batch_size: int
    dimension: int | None
    device: str | None
    base_url: str | None


def resolve_embedding_config(
    config: EmbeddingConfig,
    settings: EmbeddingSettings,
) -> ResolvedEmbeddingConfig:

    provider = "http" if settings.embedding_model_url else "local"

    return ResolvedEmbeddingConfig(
        provider=provider,
        model_name=config.embedding_model_name,
        batch_size=config.embedding_batch_size,
        dimension=config.embedding_dimension,
        device=settings.embedding_device,
        base_url=settings.embedding_model_url,
    )