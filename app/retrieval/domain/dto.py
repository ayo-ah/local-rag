from dataclasses import dataclass
from uuid import UUID
from typing import Any

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    document_section: str
    document_path: str
    score: float
    metadata: dict[str, Any]