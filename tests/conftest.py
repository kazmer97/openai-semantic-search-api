import os
import sys

from dotenv import load_dotenv
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from semantic_search_api.local_database.database_setup import setup_postgres_db
from semantic_search_api.config import get_config


sys.path.append("./")
load_dotenv()


@pytest.fixture()
async def create_test_db():
    """Creates a temporary test database for testing purposes.

    This fixture:
    1. Creates a new PostgreSQL database named 'test_vectordb'
    2. Sets up the necessary tables and schema
    3. Yields the config and database URL
    4. Cleans up by dropping the database after tests

    Returns:
        tuple: (Config object, test database URL)
    """
    test_db_name = "test_vectordb"
    database_url = os.environ["DATABASE_URL"]
    parts = database_url.rsplit("/", 1)
    master_db_url = parts[0]
    test_database_url = f"{parts[0]}/{test_db_name}"
    engine = create_async_engine(test_database_url)

    master_engine = create_async_engine(master_db_url, isolation_level="AUTOCOMMIT")
    async with master_engine.begin() as conn:
        result = await conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname=:dbname"),
            {"dbname": test_db_name},
        )
        exists = result.scalar() is not None
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{test_db_name}"'))

    config = await get_config()
    config.pg_db_async_engine = engine
    embedding_table = config.embedding_table
    try:
        await setup_postgres_db(engine, embedding_table)

        yield config, test_database_url
    finally:
        async with master_engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = '{test_db_name}' AND pid <> pg_backend_pid();
                    """
                )
            )

            await conn.execute(text(f'DROP DATABASE "{test_db_name}"'))

        await engine.dispose()
        await master_engine.dispose()


@pytest.fixture()
async def load_data_into_test_db(create_test_db):
    """Loads test data into the temporary test database.

    This fixture:
    1. Uses the temporary database created by create_test_db
    2. Loads embeddings from 'tests/small_test_data.jsonl'
    3. Stores the embeddings in the vector database

    Args:
        create_test_db: Fixture that provides the test database

    Returns:
        tuple: (Config object, test database URL)
    """
    from semantic_search_api.database.data_ingestion import (
        store_embeddings_into_vector_db,
    )

    config, test_database_url = create_test_db

    await store_embeddings_into_vector_db(
        engine=config.pg_db_async_engine,
        batch_size=10,
        concurrency=5,
        embedding_function=config.embedding_function,
        data_path="tests/small_test_data.jsonl",
        table=config.embedding_table,
    )

    yield config, test_database_url


@pytest.fixture()
def replace_env_variables(load_data_into_test_db, monkeypatch):
    """Replaces environment variables for testing.

    This fixture:
    1. Uses the test database URL from load_data_into_test_db
    2. Updates the DATABASE_URL environment variable
    3. Prevents subsequent reloading of environment variables

    Args:
        load_data_into_test_db: Fixture that provides the test database
        monkeypatch: pytest's monkeypatch fixture for modifying environment

    Returns:
        None
    """
    config, test_database_url = load_data_into_test_db
    # Load initial environment variables from .env file.
    load_dotenv()
    # Now override specific variables:
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
