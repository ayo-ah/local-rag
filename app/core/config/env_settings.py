from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path

class DatabaseSettings(BaseSettings):
    database_url: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra="ignore"


class EmbeddingSettings(BaseSettings):
    embedding_model_url: str | None
    embedding_device: str = "cpu" 
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra="ignore"

class EvalSettings(BaseSettings):
    evaluator_llm_base_url: str 
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra="ignore"


class LogSettings(BaseSettings):
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra="ignore"

class AppSettings(BaseSettings):
    experiment_config: Path = Path(
        "data/configs/responses/server_rag.json"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra="ignore"

class LLMSettings(BaseSettings):
    llm_model_path: str

    llm_url: str 
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra="ignore"


@lru_cache
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()

@lru_cache
def get_eval_settings() -> EvalSettings:
    return EvalSettings()

@lru_cache
def get_log_settings() -> LogSettings:
    return LogSettings()

@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings()

@lru_cache
def get_embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings()

@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()

