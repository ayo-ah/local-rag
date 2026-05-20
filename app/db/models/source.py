from sqlalchemy import (
    Column,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base

class Source(Base):
    __tablename__ = "sources"

    __table_args__ = (
        UniqueConstraint("type", "name", name="uq_sources_type_name"),
    )

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    type = Column(String, nullable=False)
    name = Column(String, nullable=False)
    config = Column(JSONB, nullable=True)

    documents = relationship(
        "Document",
        back_populates="source",
        cascade="all, delete-orphan",
    )