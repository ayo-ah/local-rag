from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Source


logger = get_logger(__name__)


class SourceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_type_and_name(self, type: str, name: str) -> Source | None:
        result = await self.session.execute(
            select(Source).where(
                Source.type == type,
                Source.name == name,
            )
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        type: str,
        name: str,
        config: dict | None = None,
    ) -> Source:
        source = Source(
            type=type,
            name=name,
            config=config or {},
        )

        self.session.add(source)
        await self.session.flush()

        logger.info(
            "Source_created",
            extra={
                "type": type,
                "source_name": name,
            },
        )

        return source

    async def get_or_create(
        self,
        *,
        type: str,
        name: str,
        config: dict | None = None,
    ) -> Source:
        source = await self.get_by_type_and_name(type, name)
        if source:
            return source

        return await self.create(type=type, name=name, config=config)
