import asyncio
import logging
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable
import click

import aiofiles
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine

from semantic_search_api.database.tables import EmbeddingBaseTable
from openai.types.create_embedding_response import CreateEmbeddingResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def async_load_data_from_jsonl(
    data_path: str,
) -> AsyncIterator[AsyncIterator[dict]]:
    """Creates an async context manager for reading JSONL files line by line.

    Uses asynchronous file I/O to efficiently process large JSONL files without
    loading the entire file into memory. Each line is expected to be a valid
    JSON object.

    Args:
        data_path: Path to the JSONL file to be processed.

    Yields:
        AsyncIterator[dict]: An async generator that yields individual JSON objects
            parsed from each non-empty line in the file

    Raises:
        FileNotFoundError: When the specified file path doesn't exist.
        json.JSONDecodeError: When a line contains invalid JSON data.
    """
    async with aiofiles.open(data_path, "r", encoding="utf-8") as f:

        async def generator() -> AsyncIterator[dict]:
            async for line in f:
                if line.strip():
                    yield json.loads(line)

        yield generator()


def create_string_from_row(
    row: dict, keys_to_ignore: list[str] = ["images", "videos"]
) -> str:
    """Concatenates dictionary values into a single string for embedding.

    Creates a text representation of a data row by joining key-value pairs,
    excluding specified keys that typically contain non-textual data.

    Args:
        row (dict): The input dictionary containing the data
        keys_to_ignore (list[str], optional): List of keys to exclude from the output string.
            Defaults to ["images", "videos"].

    Returns:
        str: A single string containing all included dictionary values concatenated together
    """
    return " ".join(
        f"{key}: {value}" for key, value in row.items() if key not in keys_to_ignore
    )


async def store_embeddings_into_vector_db(
    engine: AsyncEngine,
    data_path: str,
    table: EmbeddingBaseTable,
    embedding_function: Callable[[str], Awaitable[CreateEmbeddingResponse]],
    batch_size: int = 10,
    concurrency: int = 5,
    cutoff: int | None = None,
) -> int:
    """Processes data in batches and stores embeddings in a vector database.

    Implements a producer-consumer pattern to efficiently process large datasets:
    1. Reads data in batches from JSONL file
    2. Generates embeddings using the provided embedding function
    3. Stores results in PostgreSQL with vector support
    4. Manages concurrent API calls with rate limiting

    Args:
        engine: SQLAlchemy async engine for database operations.
        data_path: Path to the source JSONL file.
        table: SQLAlchemy table model for storing embeddings.
        embedding_function: Async function that generates embeddings from text.
        batch_size: Number of items to process in each batch.
        concurrency: Maximum number of concurrent embedding operations.
        cutoff: Optional maximum number of items to process.

    Returns:
        The total number of embeddings successfully stored.

    Raises:
        SQLAlchemyError: On database operation failures.
        Exception: On embedding generation failures or other processing errors.
    """
    logger.info(f"Starting embedding storage process for {data_path}")
    sem = asyncio.Semaphore(concurrency)

    async def process_and_store(rows: list[dict]) -> int:
        """Generates embeddings for a batch of rows and stores them in the database.

        Args:
            rows: List of dictionaries containing row data

        Returns:
            Number of rows processed in this batch
        """
        async with sem:
            texts = [create_string_from_row(r) for r in rows]
            response = await embedding_function(input=texts)

            # Prepare rows for insert
            values = []
            for row, embed_item in zip(rows, response.data):
                values.append({
                    "content": create_string_from_row(row),
                    "embedding": embed_item.embedding,
                    "json_content": json.dumps(row),
                })

            # Insert or update rows
            async with AsyncSession(engine) as session:
                stmt = insert(table).values(values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["data_hash"],
                    set_={
                        "content": stmt.excluded.content,
                        "embedding": stmt.excluded.embedding,
                        "json_content": stmt.excluded.json_content,
                    },
                )
                await session.execute(stmt)
                await session.commit()

            return len(rows)

    current_batch = []
    counter = 0
    total_processed = 0

    batch_queue: asyncio.Queue[list[dict]] = asyncio.Queue(maxsize=concurrency * 2)

    async def worker():
        nonlocal total_processed
        while True:
            batch = await batch_queue.get()
            if batch is None:
                batch_queue.task_done()
                break
            try:
                processed = await process_and_store(batch)
                total_processed += processed
                logger.info(f"Inserted {processed} rows, total: {total_processed}")
            except Exception as e:
                logger.error("Error processing batch", exc_info=e)
            finally:
                batch_queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

    current_batch: list[dict] = []
    async with async_load_data_from_jsonl(data_path) as data_generator:
        async for row in data_generator:
            if cutoff and counter + len(current_batch) >= cutoff:
                # Enqueue a trimmed final batch if needed.
                remaining = cutoff - counter
                if remaining > 0:
                    current_batch = current_batch[:remaining]
                    await batch_queue.put(current_batch)
                    counter += len(current_batch)
                break
            current_batch.append(row)
            if len(current_batch) >= batch_size:
                await batch_queue.put(current_batch)
                counter += len(current_batch)
                current_batch = []
    if current_batch:
        await batch_queue.put(current_batch)
        counter += len(current_batch)

    # Signal the workers that there are no more batches.
    for _ in range(concurrency):
        await batch_queue.put(None)

    # Wait until all queued batches are processed.
    await batch_queue.join()

    # Cancel workers if they're still pending.
    for w in workers:
        w.cancel()

    logger.info(f"Completed storing {total_processed} embedding records.")
    return total_processed


@click.command()
@click.option(
    "--data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the JSONL data file.",
)
@click.option(
    "--cutoff",
    default=40,
    type=int,
    help=f"Cutoff value for processing rows. default = {40}",
)
@click.option(
    "--batch_size",
    default=10,
    type=int,
    help=f"Batch size for processing data. default = {10}",
)
@click.option(
    "--concurrency",
    default=5,
    type=int,
    help=f"Number of concurrent tasks to run. default = {5}",
)
def cli(data_path, cutoff, batch_size, concurrency):
    """Command line interface for the data ingestion process.

    Provides a CLI wrapper around the embedding storage functionality,
    allowing direct execution of the ingestion process with configurable parameters.

    Args:
        data_path: Path to the input JSONL file.
        cutoff: Maximum number of rows to process.
        batch_size: Number of items to process in each batch.
        concurrency: Maximum number of concurrent operations.
    """
    from semantic_search_api.config import get_config
    from semantic_search_api.database.data_ingestion import (
        store_embeddings_into_vector_db,
    )

    async def run():
        config = await get_config()
        await store_embeddings_into_vector_db(
            engine=config.pg_db_async_engine,
            data_path=data_path,
            embedding_function=config.embedding_function,
            cutoff=cutoff,
            batch_size=batch_size,
            concurrency=concurrency,
            table=config.embedding_table,
        )

    asyncio.run(run())


if __name__ == "__main__":
    cli()
