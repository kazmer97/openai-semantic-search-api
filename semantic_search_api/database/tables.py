from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Computed, Integer, LargeBinary, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class EmbeddingBaseTable(Base):
    """Abstract base class for tables storing document embeddings.

    Defines the common schema for storing document content and their vector embeddings.
    Uses PostgreSQL-specific features for efficient document storage and retrieval:
    - JSONB for structured document storage
    - Computed SHA256 hash for deduplication
    - Unique constraint on hash for data integrity

    Attributes:
        id: Primary key for the table
        content: Text representation of the document
        json_content: Original document in JSON format
        data_hash: SHA256 hash of json_content for deduplication
    """

    __abstract__ = True
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    json_content = Column(JSONB, nullable=False)
    data_hash = Column(
        LargeBinary, Computed("digest((json_content)::text, 'sha256')", persisted=True)
    )
    __table_args__ = (
        UniqueConstraint("data_hash", name="uix_embeddings_data_hash"),
        {"extend_existing": True},
    )


def get_embedding_table(vector_size: int) -> EmbeddingBaseTable:
    """Creates a concrete embedding table class with specified vector dimensions.

    Dynamically generates a SQLAlchemy model class that inherits from EmbeddingBaseTable
    and adds a vector column of the specified size using pgvector extension.

    Args:
        vector_size: Dimension of the embedding vectors to be stored

    Returns:
        A new SQLAlchemy model class with the specified vector column
    """
    return type(
        "Embedding",
        (EmbeddingBaseTable,),
        {
            "__tablename__": "embeddings",
            "embedding": Column(Vector(vector_size)),
        },
    )
