from app.reranking.base import BaseReranker
from app.query.domain.dto import QueryContext
from app.retrieval.domain.dto import RetrievedChunk
from app.core.logging import get_logger


logger = get_logger(__name__)


class SectionBoostReranker(BaseReranker):
    def __init__(self, boost: float = 0.3):
        self.boost = boost

    async def rerank(
        self,
        results: list[RetrievedChunk],
        context: QueryContext,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:

        section_probs = context.section_probs

        if not section_probs:
            logger.debug("Section_boost_skipped_no_section_probs")
            return results[:limit] if limit else results

        logger.debug(
            "Section_boost_started",
            extra={
                "boost": self.boost,
                "section_probs": {
                    k: round(v, 4)
                    for k, v in section_probs.items()
                },
                "results_count": len(results),
            },
        )

        before_ranking = [
            {
                "rank": idx + 1,
                "chunk_id": r.chunk_id,
                "section": r.document_section,
                "base_score": round(r.score, 4),
            }
            for idx, r in enumerate(results)
        ]

        logger.debug(
            "Section_boost_before_ranking",
            extra={"ranking": before_ranking},
        )

        scored: list[RetrievedChunk] = []

        for c in results:
            base_score = c.score
            section = c.document_section

            section_prob = 0.0
            boost_value = 0.0
            final_score = base_score

            if section and section in section_probs:
                section_prob = section_probs[section]
                boost_value = self.boost * section_prob
                final_score += boost_value

            scored.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    document_section=c.document_section,
                    document_path=c.document_path,
                    score=final_score,
                    metadata={
                        **c.metadata,
                        "base_score": base_score,
                        "section_prob": section_prob,
                        "boost_value": boost_value,
                    },
                )
            )

        ranked = sorted(
            scored,
            key=lambda c: c.score,
            reverse=True,
        )

        after_ranking = []

        for idx, r in enumerate(ranked):
            previous_rank = next(
                i + 1
                for i, original in enumerate(results)
                if original.chunk_id == r.chunk_id
            )

            after_ranking.append(
                {
                    "old_rank": previous_rank,
                    "new_rank": idx + 1,
                    "rank_delta": previous_rank - (idx + 1),
                    "chunk_id": r.chunk_id,
                    "section": r.document_section,
                    "base_score": round(
                        r.metadata.get("base_score", 0.0),
                        4,
                    ),
                    "section_prob": round(
                        r.metadata.get("section_prob", 0.0),
                        4,
                    ),
                    "boost_value": round(
                        r.metadata.get("boost_value", 0.0),
                        4,
                    ),
                    "final_score": round(r.score, 4),
                }
            )

        logger.debug(
            "Section_boost_after_ranking",
            extra={"ranking": after_ranking},
        )

        moved_chunks = [
            r for r in after_ranking
            if r["rank_delta"] != 0
        ]

        if moved_chunks:
            logger.info(
                "Section_boost_changed_ranking",
                extra={
                    "moved_chunks": moved_chunks,
                },
            )
        else:
            logger.info("Section_boost_no_ranking_changes")

        return ranked[:limit] if limit else ranked