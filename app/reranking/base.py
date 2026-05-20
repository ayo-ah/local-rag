from typing import Protocol
from collections.abc import Sequence
from app.retrieval.domain.dto import RetrievedChunk
from app.query.domain.dto import QueryContext

class BaseReranker(Protocol):
    async def rerank(
        self,
        results: list[RetrievedChunk],
        context: QueryContext,
        limit: int | None = None,
    ) -> Sequence[RetrievedChunk]:
        ...
