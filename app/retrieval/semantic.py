from collections.abc import Sequence

from app.embeddings.protocol import EmbeddingService
from app.db.repositories.embeddings import EmbeddingRepository
from app.retrieval.domain.dto import RetrievedChunk
from app.retrieval.base import BaseRetriever
from app.core.logging import get_logger


logger = get_logger(__name__)


class SemanticRetriever(BaseRetriever):
    name = "semantic"
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        embedding_repo: EmbeddingRepository,
        limit: int,
    ):
        self.embedding_service = embedding_service
        self.embedding_repo = embedding_repo
        self.limit = limit

    async def retrieve(
        self,
        query: str,
    ) -> Sequence[RetrievedChunk]:

        logger.debug(
            "Semantic_retrieval_started",
            extra={
                "model_name": self.embedding_service.model_name,
                "limit": self.limit,
            },
        )

        vector = await self.embedding_service.embed_query(query)

        logger.debug(
            "Query_embedded",
            extra={"dimension": len(vector)},
        )

        results = await self.embedding_repo.semantic_search(
            vector=vector,
            top_k=self.limit,
            model_name=self.embedding_service.model_name,
        )

        logger.debug(
            "Semantic_retrieval_finished",
            extra={"results_count": len(results)},
        )

        return results
