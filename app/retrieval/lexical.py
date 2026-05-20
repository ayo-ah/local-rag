from collections.abc import Sequence

from app.retrieval.base import BaseRetriever
from app.db.repositories.chunks import ChunkRepository
from app.retrieval.domain.dto import RetrievedChunk
from app.core.logging import get_logger


logger = get_logger(__name__)


class LexicalRetriever(BaseRetriever):
    name = "lexical"
    
    def __init__(
        self,
        chunk_repo: ChunkRepository,
        limit: int,
        language: str = "russian",
        
    ):
        self.language = language
        self.chunk_repo = chunk_repo
        self.limit = limit
    async def retrieve(
        self,
        query: str,
    ) -> Sequence[RetrievedChunk]:

        logger.debug(
            "Lexical_retrieval_started",
            extra={
                "language": self.language,
                "limit": self.limit,
            },
        )

        results = await self.chunk_repo.lexical_search(
            query=query,
            top_k=self.limit,
            language=self.language,
        )

        logger.debug(
            "Lexical_retrieval_finished",
            extra={"results_count": len(results)},
        )

        return results
