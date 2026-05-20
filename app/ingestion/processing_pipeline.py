from collections.abc import Sequence

from app.ingestion.extractors.base import BaseExtractor
from app.ingestion.domain.dto import ExtractedDocument
from app.ingestion.chunkers.base import TextChunker
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessingPipeline:
    def __init__(
        self,
        extractors: Sequence[BaseExtractor],
        chunker: TextChunker,
    ):
        self.extractors = extractors
        self.chunker = chunker

    def _resolve_extractor(self, mime_type: str) -> BaseExtractor:
        for extractor in self.extractors:
            if extractor.supports(mime_type):
                return extractor

        logger.error(
            "No_extractor_found_for_mime_type",
            extra={
                "mime_type": mime_type,
                "available_extractors": [
                    extractor.__class__.__name__
                    for extractor in self.extractors
                ],
            },
        )

        raise ValueError(f"No extractor for mime_type={mime_type}")

    async def run(self, data: bytes, mime_type: str):
        extractor = self._resolve_extractor(mime_type)
        extracted: ExtractedDocument = await extractor.extract(data)

        chunks = self.chunker.chunk(extracted.text)

        logger.debug(
            "Ingestion_pipeline_completed",
            extra={
                "mime_type": mime_type,
                "chunks_count": len(chunks),
            },
        )

        return extracted, chunks
