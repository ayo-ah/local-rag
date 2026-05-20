from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import text, bindparam
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector

from app.core.logging import get_logger
from app.retrieval.domain.dto import RetrievedChunk
from app.db.models import Embedding


logger = get_logger(__name__)

class EmbeddingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_embeddings(
        self,
        *,
        chunk_ids: list[UUID],
        vectors: list[list[float]],
        model_name: str,
    ) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors length mismatch")

        rows = [
            {
                "chunk_id": chunk_id,
                "model_name": model_name,
                "embedding": vector,
            }
            for chunk_id, vector in zip(chunk_ids, vectors)
        ]

        sql = text("""
            INSERT INTO embeddings (chunk_id, model_name, embedding)
            VALUES (:chunk_id, :model_name, :embedding)
        """).bindparams(
            bindparam("embedding", type_=Vector(len(vectors[0])))
        )

        await self.session.execute(sql, rows)
    
    async def semantic_search(
        self,
        *,
        vector: list[float],
        top_k: int,
        model_name: str,
    ) -> Sequence[RetrievedChunk]:
        sql = text("""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    d.external_id,
                    d.attributes->>'section' AS section,
                    1 - (e.embedding <=> :vector) AS score
                FROM embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE e.model_name = :model_name
                ORDER BY e.embedding <=> :vector
                LIMIT :top_k
        """).bindparams(bindparam("vector", type_=Vector(768)))

        result = await self.session.execute(
            sql,
            {
                "vector": vector,
                "top_k": top_k,
                "model_name": model_name,
            }
        )

        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                document_section=row.section,
                document_path=row.external_id,
                score=row.score,
                metadata={"retrieval_method": "cosine"},
            )
            for row in result
        ]