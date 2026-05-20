from uuid import uuid4
from typing import List

from app.evaluation.domain.dto import EvalChunk


class MockChunkRepository:

    async def get_chunks_for_evaluation(self, source_id, limit: int):

        chunks: List[EvalChunk] = []

        for i in range(limit):
            chunks.append(
                EvalChunk(
                    content=f"Chunk content {i}",
                    document_id=str(uuid4()),
                    chunk_id=str(uuid4()),
                    chunk_index=i,
                    metadata={"source_id": source_id}
                )
            )

        return chunks
