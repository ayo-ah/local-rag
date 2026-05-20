from sqlalchemy import (
    Column,
    Text,
    Integer,
    ForeignKey,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import relationship

from app.db.base import Base

class Chunk(Base):
    __tablename__ = "chunks"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_chunks_document_index",
        ),
    )

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    tsv = Column(TSVECTOR)
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship(
        "Embedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )