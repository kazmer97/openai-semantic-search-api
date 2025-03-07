"""Database setup module for PostgreSQL vector database initialization.

This module handles the creation and configuration of PostgreSQL database
for vector search capabilities. It ensures required extensions are installed
and manages the embeddings table schema.

Key responsibilities:
- Installing required PostgreSQL extensions (pgcrypto, vector)
- Managing table schemas (creation, deletion)
- Verifying database setup
"""

import logging
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine
from semantic_search_api.config import get_config
from semantic_search_api.database.tables import EmbeddingBaseTable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def setup_postgres_db(engine: AsyncEngine, embedding_table: EmbeddingBaseTable):
    """Sets up PostgreSQL database with vector search capabilities.

    Performs complete database initialization:
    1. Installs required PostgreSQL extensions
    2. Drops existing tables (if any)
    3. Creates new tables with vector support
    4. Verifies table creation

    Args:
        engine: SQLAlchemy async engine for database operations
        embedding_table: SQLAlchemy model defining the table schema

    Raises:
        SQLAlchemyError: If database operations fail
    """

    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: sync_conn.execute(
                text("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            )
        )
        await conn.run_sync(
            lambda sync_conn: sync_conn.execute(
                text("CREATE EXTENSION IF NOT EXISTS vector;")
            )
        )
        await conn.run_sync(
            lambda sync_conn: embedding_table.metadata.drop_all(sync_conn)
        )
        logger.info("Tables dropped (if existed).")

        await conn.run_sync(
            lambda sync_conn: embedding_table.metadata.create_all(sync_conn)
        )
        logger.info("Tables created successfully.")

        for table_name in embedding_table.metadata.tables.keys():
            exists = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).has_table(table_name)
            )
            logger.info(f"Table '{table_name}' exists: {exists}")


async def main():
    config = await get_config()
    await setup_postgres_db(
        engine=config.pg_db_async_engine, embedding_table=config.embedding_table
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
