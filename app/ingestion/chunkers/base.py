from typing import Protocol
from ..domain.dto import TextChunk


class TextChunker(Protocol):
    chunk_size: int
    overlap: int
    
    def chunk(self, text: str) -> list[TextChunk]:
        ...
