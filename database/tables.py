from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeMeta

Base = declarative_base()


def create_embedding_table_model(vector_size: int) -> DeclarativeMeta:
    class Embedding(Base):
        __tablename__ = "embeddings"
        id = Column(Integer, primary_key=True)
        content = Column(Text, nullable=False)
        embedding = Column(Vector(vector_size))

    return Embedding
