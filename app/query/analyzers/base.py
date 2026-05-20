from typing import Protocol
from app.query.domain.dto import QueryContext

class QueryAnalyzer(Protocol):
    async def analyze(self, context: QueryContext) -> None:
        ...
