from uuid import UUID
from dataclasses import dataclass, field, asdict
from typing import  Any
from app.retrieval.domain.dto import RetrievedChunk
from app.core.config.experiment_config import EvaluationConfig
@dataclass
class EvalChunk:
    chunk_id: UUID
    document_id: UUID
    document_path: str
    chunk_index: int
    content: str
    metadata: dict

@dataclass
class EvalSample:
    question_id: str
    question: str
    question_type: str
    reference_answer: str
    score: float
    reasoning: str
    source: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class RAGResponse:
    question_id: str
    question: str
    answer: str
    sources: list[str]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    hit_rate: float


@dataclass
class GenerationScore:
    question_id: str
    score: float
    reasoning: str


@dataclass
class GenerationMetrics:
    mean_score: float
    scores: list[GenerationScore]


@dataclass
class EvalBundle:
    config: EvaluationConfig
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    responses: list[RAGResponse]