from dataclasses import dataclass

@dataclass
class ExtractedDocument:
    text: str
    metadata: dict
    mime_type: str
    page_map: dict[int, str] | None = None


@dataclass(frozen=True)
class TextChunk:
    content: str
    index: int



