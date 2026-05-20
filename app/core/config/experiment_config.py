from pydantic import BaseModel
from pydantic import Field
from pathlib import Path

class EmbeddingConfig(BaseModel):
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_batch_size: int = 32
    embedding_dimension: int = 768

class IngestionConfig(BaseModel):
    chunk_size: int = 200
    chunk_overlap: int = 20
    source_path: Path = Path("data/files/raw")
    source_type: str = "local_fs"
    source_name: str = "local"

class RetrievalConfig(BaseModel):
    retriever_type: str = "semantic"
    per_retriever_limits: dict[str, int] = Field(
        default_factory=lambda: {
            "semantic": 20,
            "lexical": 5,
        }
    )
    top_k: int = 5

class RerankerConfig(BaseModel):
    reranker_type: str | None = None
    reranker_boost: float | None = None
    section_classifier_path: Path | None = Path("app/classification/artifacts/section_classifier_v1.joblib")
    section_threshold: float | None = 0.6

class PromptConfig(BaseModel):
    task: str
    version: str
    root: Path = Path("app/prompting/templates")

class LLMConfig(BaseModel):
    llm_max_context: int = 4096
    llm_max_tokens: int = 256
    llm_temperature: float = 0.2
    llm_name: str = "qwen2.5-0.5b-instruct-q4_k_m"
    llm_stop: list[str] = Field(default_factory=lambda: ["\n\n"])

class ExperimentConfig(BaseModel):
    experiment_name: str
    
    dataset_path: Path = Path("data/experiments/references/eval/dataset.json")
    responses_dir: Path = Path("data/experiments/responses")

    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    llm: LLMConfig
    prompt: PromptConfig
    embedding_model: EmbeddingConfig

class EvaluationConfig(BaseModel):
    experiment_name: str

    evaluator: str
    evaluator_prompt: PromptConfig
    dataset_path: Path
    responses_path: Path
    evaluation_dir: Path
