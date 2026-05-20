from typing import Protocol
from collections.abc import Sequence

from .domain.dto import RetrievedChunk


class BaseRetriever(Protocol):
    name: str
    limit: int
    async def retrieve(
        self,
        query: str,
    ) -> Sequence[RetrievedChunk]:
        ...
