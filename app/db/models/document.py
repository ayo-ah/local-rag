from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    external_id = Column(String, nullable=True)
    title = Column(String, nullable=True)
    content_hash = Column(String, nullable=False, unique=True, index=True)
    attributes = Column(JSONB, nullable=True)

    source = relationship("Source", back_populates="documents")
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

