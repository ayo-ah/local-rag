from pydantic import BaseModel
from pydantic import Field
from pathlib import Path
from uuid import UUID

from .experiment_config import LLMConfig, PromptConfig

class DatasetGeneratorConfig(BaseModel):
    output_dir: Path = Path("data/experiments/references/eval")
    dataset_name: str = "dataset.json"
    llm: LLMConfig
    prompt: PromptConfig
    questions_per_source: int = 5
    limit_chunks: int = 5
    source_id: list[UUID] = Field(default_factory=list)