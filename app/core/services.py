from functools import cached_property

from app.core.config.env_settings import (
    LLMSettings,
)
from app.core.config.experiment_config import (
    LLMConfig,
)
from app.core.config.embedding_config import ResolvedEmbeddingConfig
from app.embeddings.factory import EmbeddingFactory
from app.llm.http import LlamaCppHttpLLMService


class AppServices:
    def __init__(
        self,
        llm_settings: LLMSettings,
        embedding_config: ResolvedEmbeddingConfig,
        llm_config: LLMConfig,
    ):
        self.llm_settings = llm_settings
        self.embedding_config = embedding_config
        self.llm_config = llm_config

    @cached_property
    def embedding_service(self):
        return EmbeddingFactory.create(self.embedding_config)

    @cached_property
    def llm_service(self):
        return LlamaCppHttpLLMService(
            base_url=self.llm_settings.llm_url,
            max_tokens=self.llm_config.llm_max_tokens,
            temperature=self.llm_config.llm_temperature,
            stop=self.llm_config.llm_stop,
        )
