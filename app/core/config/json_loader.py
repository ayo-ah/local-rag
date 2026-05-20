from pathlib import Path

from app.utils.json_io import load_json
from .experiment_config import ExperimentConfig, EvaluationConfig
from .dataset_generation_config import DatasetGeneratorConfig

def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = load_json(path)
    return ExperimentConfig(**raw)

def load_evaluation_config(path: Path) -> EvaluationConfig:
    raw = load_json(path)
    return EvaluationConfig(**raw)

def load_dataset_generator_config(path: Path) -> DatasetGeneratorConfig:
    raw = load_json(path)
    return DatasetGeneratorConfig(**raw)