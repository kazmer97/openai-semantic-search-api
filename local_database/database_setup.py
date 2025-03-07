import os

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, Text, create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from database.tables import create_embedding_table_model

load_dotenv()


VECTOR_SIZE = int(os.environ["VECTOR_SIZE"])
DATABASE_URL = os.environ["DATABASE_URL"]

Base = declarative_base()

Embedding = create_embedding_table_model(VECTOR_SIZE)

if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    # Create an inspector to check for the table
    inspector = inspect(engine)

    if inspector.has_table("embeddings"):
        # Drop the table if it exists
        Embedding.__table__.drop(engine)
        print("Table 'embeddings' dropped.")
    else:
        print("Table 'embeddings' does not exist.")

    print("creating table")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    print("embeddings table exists: " + str(inspector.has_table("embeddings")))
