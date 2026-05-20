from typing import Protocol

from ..domain.dto import ExtractedDocument

class BaseExtractor(Protocol):
    def supports(self, mime_type: str) -> bool:
        ...

    async def extract(self, data: bytes) -> ExtractedDocument:
        ...
