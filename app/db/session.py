from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config.env_settings import get_database_settings
from app.core.logging import get_logger


logger = get_logger(__name__)

db_settings = get_database_settings()

engine = create_async_engine(
    db_settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

logger.info("Database_engine_initialized")

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
