from collections.abc import Iterable, Sequence
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Chunk
from app.ingestion.domain.dto import TextChunk
from app.retrieval.domain.dto import RetrievedChunk
from app.evaluation.domain.dto import EvalChunk

logger = get_logger(__name__)


class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_chunks(
        self,
        *,
        document_id: UUID,
        chunks: Iterable[TextChunk],
    ) -> list[UUID]:

        chunks = list(chunks)

        if not chunks:
            return []

        values_sql = []
        params = {}

        for i, chunk in enumerate(chunks):
            values_sql.append(
                f"(:document_id_{i}, :chunk_index_{i}, :content_{i})"
            )

            params[f"document_id_{i}"] = document_id
            params[f"chunk_index_{i}"] = chunk.index
            params[f"content_{i}"] = chunk.content

        sql = text(f"""
            INSERT INTO chunks (document_id, chunk_index, content)
            VALUES {", ".join(values_sql)}
            RETURNING id
        """)

        result = await self.session.execute(sql, params)

        chunk_ids = [row.id for row in result]

        logger.info(
            "Chunks_added",
            extra={
                "document_id": str(document_id),
                "chunks_count": len(chunk_ids),
            },
        )

        return chunk_ids

    async def list_chunks(
        self,
        document_id: UUID,
    ) -> Sequence[Chunk]:
        stmt = select(Chunk).where(Chunk.document_id == document_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    def _to_or_query(self, query: str) -> str:
        tokens = [t for t in query.split() if t.strip()]
        return " OR ".join(tokens)


    async def lexical_search(
        self,
        query: str,
        top_k: int,
        language: str = "russian",
    ) -> Sequence[RetrievedChunk]:
        logger.debug(
            "Lexical_search",
            extra={
                "query": query,
                "top_k": top_k,
                "language": language,
            },
        )

        query_or = self._to_or_query(query)

        sql = text("""
            WITH q AS (
                SELECT
                    websearch_to_tsquery(:lang, :query)      AS q_and,
                    websearch_to_tsquery(:lang, :query_or)   AS q_or
            )
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content,
                d.external_id,
                d.attributes->>'section' AS section,

                (
                    ts_rank_cd(c.tsv, q.q_and, 32) * 0.7 +
                    ts_rank_cd(c.tsv, q.q_or, 32)  * 0.3
                ) AS score

            FROM chunks c
            JOIN documents d ON d.id = c.document_id,
                q

            WHERE
                c.tsv @@ q.q_or

            ORDER BY score DESC
            LIMIT :top_k
        """)

        result = await self.session.execute(
            sql,
            {
                "query": query,
                "query_or": query_or,
                "top_k": top_k,
                "lang": language,
            },
        )

        rows = result.fetchall()

        logger.info(
            "Lexical_search_completed",
            extra={
                "returned_chunks": len(rows),
                "top_k": top_k,
            },
        )

        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                document_section=row.section,
                document_path=row.external_id,
                score=float(row.score) if row.score is not None else 0.0,
                metadata={"retrieval_method": "lexical"},
            )
            for row in rows
        ]