from pathlib import Path

from app.ingestion.processing_pipeline import DocumentProcessingPipeline
from app.db.models import Document
from app.db.repositories.documents import DocumentRepository
from app.db.repositories.embeddings import EmbeddingRepository
from app.db.repositories.chunks import ChunkRepository
from app.embeddings.protocol import EmbeddingService
from app.utils.hashing import hash_bytes
from app.core.logging import get_logger

logger = get_logger(__name__)


class IngestService:
    def __init__(
        self,
        pipeline: DocumentProcessingPipeline,
        document_repo: DocumentRepository,
        chunks_repo: ChunkRepository,
        embeddings_repo: EmbeddingRepository,
        embedding_service: EmbeddingService,
    ):
        self.pipeline = pipeline
        self.document_repo = document_repo
        self.chunks_repo = chunks_repo
        self.embeddings_repo = embeddings_repo
        self.embedding_service = embedding_service

    async def ingest(
        self,
        *,
        source_id,
        reference: str,
        data: bytes,
        mime_type: str,
        metadata: dict,
    ) -> Document:
        logger.info(
            "Ingestion_started",
            extra={
                "source_id": str(source_id),
                "reference": reference,
                "mime_type": mime_type,
                "size_bytes": len(data),
            },
        )

        try:
            content_hash = hash_bytes(data)
            existing = await self.document_repo.get_by_content_hash(content_hash)
            if existing:
                logger.info(
                    "Document_already_exists_skipped",
                    extra={
                        "document_id": str(existing.id),
                        "source_id": str(source_id),
                        "reference": reference,
                        "content_hash": content_hash,
                    },
                )
                return existing
            extracted, chunks = await self.pipeline.run(data, mime_type)
            document_section = self._detect_section(reference)
            extracted.metadata['section'] = document_section
            document = await self.document_repo.create(
                source_id=source_id,
                external_id=reference,
                title=metadata.get("title"),
                content_hash=content_hash,
                metadata=extracted.metadata,
            )

            chunk_ids = await self.chunks_repo.add_chunks(
                document_id=document.id,
                chunks=chunks,
            )

            vectors = await self.embedding_service.embed_documents(
                [c.content for c in chunks]
            )

            await self.embeddings_repo.add_embeddings(
                chunk_ids=chunk_ids,
                vectors=vectors,
                model_name=self.embedding_service.model_name,
            )

        except Exception:
            logger.exception(
                "Ingestion_failed",
                extra={
                    "source_id": str(source_id),
                    "reference": reference,
                },
            )
            raise

        logger.info(
            "Ingestion_completed",
            extra={
                "document_id": str(document.id),
                "chunks_count": len(chunk_ids),
                "embeddings_count": len(vectors),
                "embedding_model": self.embedding_service.model_name,
            },
        )

        return document
    
    def _detect_section(self, reference: str) -> str:
        p = Path(reference)
        if p.parent.name:
            return p.parent.name
        return "root"
 