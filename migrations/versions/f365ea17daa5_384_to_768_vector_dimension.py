"""384 to 768 vector dimension

Revision ID: f365ea17daa5
Revises: 1c3500c62171
Create Date: 2026-03-03 22:51:45.093632

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'f365ea17daa5'
down_revision: Union[str, Sequence[str], None] = '1c3500c62171'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_embeddings_embedding_hnsw", table_name="embeddings")

    op.drop_column("embeddings", "embedding")

    op.add_column(
        "embeddings",
        sa.Column("embedding", Vector(768), nullable=False),
    )
    
    op.create_index(
        "ix_embeddings_embedding_hnsw",
        "embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_embeddings_embedding_hnsw", table_name="embeddings")
    op.drop_column("embeddings", "embedding")

    op.add_column(
        "embeddings",
        sa.Column("embedding", Vector(384), nullable=False),
    )

    op.create_index(
        "ix_embeddings_embedding_hnsw",
        "embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
