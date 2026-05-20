import mimetypes
from sqlalchemy.ext.asyncio import AsyncSession


from app.storage.base import BaseStorage
from app.ingestion.chunkers.langchain_recursive import LangChainRecursiveChunker
from app.ingestion.extractors.pdf import PdfExtractor
from app.ingestion.processing_pipeline import DocumentProcessingPipeline
from app.ingestion.ingest_service import IngestService
from app.ingestion.batch_ingest import BatchIngestService
from app.core.config.experiment_config import IngestionConfig
from app.db.repositories.sources import SourceRepository
from app.db.repositories.documents import DocumentRepository
from app.db.repositories.chunks import ChunkRepository
from app.db.repositories.embeddings import EmbeddingRepository
from app.embeddings.protocol import EmbeddingService

async def ingest_documents(
    session: AsyncSession,
    ingestion_config: IngestionConfig,
    embedding_service: EmbeddingService,
    storage: BaseStorage,
    
):

    pipeline = DocumentProcessingPipeline(
        extractors=[PdfExtractor()],
        chunker=LangChainRecursiveChunker(
            chunk_size=ingestion_config.chunk_size,
            overlap=ingestion_config.chunk_overlap,
        ),
    )

    doc_repo = DocumentRepository(session)
    chunk_repo = ChunkRepository(session)
    emb_repo = EmbeddingRepository(session)
    src_repo = SourceRepository(session)

    ingest_service = IngestService(
        pipeline=pipeline,
        document_repo=doc_repo,
        chunks_repo=chunk_repo,
        embeddings_repo=emb_repo,
        embedding_service=embedding_service,
    )

    orchestrator = BatchIngestService(
        storage=storage,
        ingest_service=ingest_service,
        source_repo=src_repo,
    )

    await orchestrator.ingest_batch(
        source_type=ingestion_config.source_type,
        source_name=ingestion_config.source_name,
        mime_detector=detect_mime
    )

    await session.commit()
        
def detect_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"

