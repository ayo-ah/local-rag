import asyncio
from collections import defaultdict
from collections.abc import Sequence

from app.retrieval.base import BaseRetriever
from app.retrieval.domain.dto import RetrievedChunk
from app.core.logging import get_logger


logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    name = "hybrid"
    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        normalize_scores: bool = True,
    ):
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one retriever")

        self.retrievers = retrievers
        self.normalize_scores = normalize_scores

    async def retrieve(
        self,
        query: str,
    ) -> Sequence[RetrievedChunk]:

        logger.debug(
            "Hybrid_retrieval_started",
            extra={
                "retrievers": len(self.retrievers),
            },
        )

        tasks = [ r.retrieve(query) for r in self.retrievers ]

        results_per_retriever = await asyncio.gather(*tasks)

        counts = [len(r) for r in results_per_retriever]
        logger.debug(
            "Hybrid_retrievers_returned_results",
            extra={"per_retriever_counts": counts},
        )

        all_results: list[RetrievedChunk] = [
            r for group in results_per_retriever for r in group
        ]

        if not all_results:
            return []

        if self.normalize_scores:
            all_results = self._normalize(all_results)
            logger.debug("Scores_normalized")

        before = len(all_results)
        deduped = self._deduplicate(all_results)
        after = len(deduped)

        if after < before:
            logger.debug(
                "Hybrid_deduplication_applied",
                extra={
                    "before": before,
                    "after": after,
                },
            )

        return deduped

    def _normalize(
        self,
        results: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        grouped = defaultdict(list)

        for r in results:
            source = r.metadata.get("source", "unknown")
            grouped[source].append(r)

        normalized: list[RetrievedChunk] = []

        for source, group in grouped.items():
            scores = [r.score for r in group]
            min_s, max_s = min(scores), max(scores)

            for r in group:
                score = r.score
                if max_s > min_s:
                    score = (score - min_s) / (max_s - min_s)

                normalized.append(
                    RetrievedChunk(
                        chunk_id=r.chunk_id,
                        document_id=r.document_id,
                        content=r.content,
                        document_section=r.document_section,
                        document_path=r.document_path,
                        score=score,
                        metadata={
                            **r.metadata,
                            "normalized": True,
                        },
                    )
                )

        return normalized

    def _deduplicate(
        self,
        results: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:

        best_by_chunk: dict[str, RetrievedChunk] = {}

        for r in results:
            existing = best_by_chunk.get(r.chunk_id)
            if not existing or r.score > existing.score:
                best_by_chunk[r.chunk_id] = r

        return list(best_by_chunk.values())
