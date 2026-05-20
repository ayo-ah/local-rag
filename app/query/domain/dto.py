from dataclasses import dataclass, field

from app.retrieval.domain.dto import RetrievedChunk
@dataclass
class Query:
    text: str
    top_k: int = 5

@dataclass
class QueryContext:
    query: Query
    section_probs: dict[str, float] = field(default_factory=dict)
    intent: str | None = None
    features: dict[str, float] = field(default_factory=dict)

@dataclass
class QueryAgentResponse:
    answer: str
    sources: list[RetrievedChunk]
