from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db.base import Base

class Embedding(Base):
    __tablename__ = "embeddings"

    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "model_name",
            name="uq_embeddings_chunk_model",
        ),
    )

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    chunk_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_name = Column(String, nullable=False, index=True)
    embedding = Column(Vector(768), nullable=False)

    chunk = relationship("Chunk", back_populates="embeddings")