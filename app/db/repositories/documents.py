from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Document


logger = get_logger(__name__)


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_content_hash(self, content_hash: str) -> Document | None:
        stmt = select(Document).where(Document.content_hash == content_hash)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, document_id: UUID) -> Document | None:
        stmt = select(Document).where(Document.id == document_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        source_id: UUID,
        external_id: str | None,
        title: str | None,
        content_hash: str,
        metadata: dict | None,
    ) -> Document:
        document = Document(
            source_id=source_id,
            external_id=external_id,
            title=title,
            content_hash=content_hash,
            attributes=metadata,
        )

        self.session.add(document)
        await self.session.flush()

        logger.info(
            "Document_created",
            extra={
                "document_id": str(document.id),
                "source_id": str(source_id),
                "external_id": external_id,
                "content_hash": content_hash,
            },
        )

        return document

    async def delete_document(self, document_id: UUID) -> None:
        stmt = delete(Document).where(Document.id == document_id)
        await self.session.execute(stmt)

        logger.warning(
            "Document_deleted",
            extra={
                "document_id": str(document_id),
            },
        )

    async def get_external_ids(
        self,
        ids: set[UUID],
        ) -> dict[UUID, str]:
        if not ids:
            return{}
        stmt = select(
            Document.id,
            Document.external_id,
        ).where(Document.id.in_(ids))
        result = await self.session.execute(stmt)
        return {
            row.id: row.external_id
            for row in result.fetchall()
        }
