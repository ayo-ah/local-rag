from pathlib import Path

from dataclasses import asdict, is_dataclass

from app.core.config.experiment_config import ExperimentConfig
from app.utils.json_io import save_json, load_json
from app.evaluation.domain.dto import EvalBundle
from app.evaluation.domain.dto import RAGResponse
from app.retrieval.domain.dto import RetrievedChunk
from app.utils.hashing import hash_config      
    
def write_inference_artifact(responses: list[RAGResponse], config: ExperimentConfig):

    config_hash = hash_config(config)
    name = f"{config.experiment_name}_{config_hash}.json"

    path = Path(config.responses_dir) / name
    serialized_responses = [
        asdict(r) if is_dataclass(r) else r
        for r in responses
    ]
    payload = {
        "experiment_config": config,
        "responses": serialized_responses,
    }

    save_json(path, payload)

    return path

def write_evaluation_artifact(bundle: EvalBundle):

    config_hash = hash_config(bundle.config)
    name = f"{bundle.config.experiment_name}_{config_hash}.json"

    path = Path(bundle.config.evaluation_dir) / name

    save_json(path, asdict(bundle))

    return path

def read_inference_artifact(path: Path) -> list[RAGResponse]:

    raw = load_json(path)["responses"]

    return [
        RAGResponse(
            question_id=i["question_id"],
            question=i["question"],
            answer=i["answer"],
            sources= i["sources"],
        )
        for i in raw
    ]
