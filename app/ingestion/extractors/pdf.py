import io
from pypdf import PdfReader

from .base import BaseExtractor, ExtractedDocument
from app.core.logging import get_logger


logger = get_logger(__name__)


class PdfExtractor(BaseExtractor):
    def supports(self, mime_type: str) -> bool:
        return mime_type == "application/pdf"

    async def extract(self, data: bytes) -> ExtractedDocument:
        logger.info(
            "Starting_PDF_extraction",
            extra={
                "size_bytes": len(data),
                "mime_type": "application/pdf",
            },
        )

        try:
            reader = PdfReader(io.BytesIO(data))
            pages: dict[int, str] = {}

            for i, page in enumerate(reader.pages):
                pages[i + 1] = page.extract_text() or ""

        except Exception:
            logger.exception("Failed_to_extract PDF")
            raise

        non_empty_pages = sum(1 for text in pages.values() if text.strip())
        text = "\n".join(pages.values())

        logger.info(
            "PDF_extracted",
            extra={
                "pages_total": len(pages),
                "pages_with_text": non_empty_pages,
                "text_length": len(text),
            },
        )

        return ExtractedDocument(
            text=text,
            metadata={"pages": len(pages)},
            mime_type="application/pdf",
            page_map=pages,
        )
