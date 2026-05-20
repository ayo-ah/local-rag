from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import get_logger
from ..domain.dto import TextChunk


logger = get_logger(__name__)


class LangChainRecursiveChunker:
    def __init__(
        self,
        chunk_size: int,
        overlap: int,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

        logger.info(
            "Initializing_text_chunker",
            extra={
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

    def chunk(self, text: str) -> list[TextChunk]:
        texts = self.splitter.split_text(text)

        logger.info(
            "Text_chunked",
            extra={
                "text_length": len(text),
                "chunks_count": len(texts),
            },
        )

        chunks: list[TextChunk] = []

        for i, t in enumerate(texts):
            chunks.append(
                TextChunk(
                    content=t,
                    index=i,
                )
            )

        logger.debug(
            "Chunk_sizes",
            extra={
                "sizes": [len(c.content) for c in chunks],
            },
        )

        return chunks
