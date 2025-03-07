"""
Configuration module for the semantic search API.
This module handles all configuration settings, environment variables,
and initialization of core components like OpenAI client and database connections.
"""

import asyncio
from dataclasses import dataclass
import logging
import os
from typing import Awaitable, Callable

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from openai import AsyncOpenAI, NotFoundError
from openai.types.create_embedding_response import CreateEmbeddingResponse

from semantic_search_api.database.tables import (
    EmbeddingBaseTable,
    get_embedding_table,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

load_dotenv()


class BadConfigValue(Exception):
    """Raised when configuration values are invalid or incompatible with the application requirements."""

    pass


# Environment variable configurations
OPENAI_API_ORGANIZATION = os.environ[
    "OPENAI_API_ORGANIZATION"
]  # OpenAI organization identifier
OPENAI_API_PROJECT_ID = os.environ["OPENAI_API_PROJECT_ID"]  # OpenAI project identifier
VECTOR_SIZE = int(os.environ["VECTOR_SIZE"])  # Dimension size for embeddings
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]  # OpenAI embedding model name
RE_RANK_MODEL = os.environ["RE_RANK_MODEL"]  # Model used for re-ranking results
QUERY_REFINEMENT_MODEL = os.environ[
    "QUERY_REFINEMENT_MODEL"
]  # Model used for query refinement
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_DB = os.environ["POSTGRES_DB"]
DATABASE_HOST = os.environ["DATABASE_HOST"]


def create_async_openai_client():
    """
    Creates an authenticated async OpenAI client instance.

    Returns:
        AsyncOpenAI: Configured OpenAI client with retry logic
    """
    return AsyncOpenAI(
        organization=OPENAI_API_ORGANIZATION,
        project=OPENAI_API_PROJECT_ID,
        max_retries=5,
    )


def create_async_db_engine():
    """
    Creates an async SQLAlchemy engine for database operations.

    Returns:
        AsyncEngine: Configured PostgreSQL async engine
    """
    database_url = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DATABASE_HOST}:5432/vectordb"

    return create_async_engine(database_url)


def create_embedding_table():
    """
    Creates a database table schema for storing embeddings.

    Returns:
        EmbeddingBaseTable: SQLAlchemy table definition for embeddings
    """
    return get_embedding_table(VECTOR_SIZE)


def create_embedding_function(
    embedding_model: str, vector_size: int, oai_client: AsyncOpenAI
) -> Callable[[str | list[str]], Awaitable[CreateEmbeddingResponse]]:
    """
    Creates a function that generates embeddings using OpenAI's API.

    Args:
        embedding_model (str): Name of the OpenAI embedding model
        vector_size (int): Desired dimension size of the embeddings
        oai_client (AsyncOpenAI): Authenticated OpenAI client

    Returns:
        Callable: Async function that takes text input and returns embeddings
    """

    async def embedding_fn(input: str | list[str]) -> CreateEmbeddingResponse:
        """
        Inner function that calls OpenAI's embedding API.

        Args:
            input (str | list[str]): Text to generate embeddings for

        Returns:
            CreateEmbeddingResponse: OpenAI API response containing embeddings
        """
        response = await oai_client.embeddings.create(
            input=input,
            model=embedding_model,
            encoding_format="float",
            dimensions=vector_size,
        )
        return response

    return embedding_fn


@dataclass
class Config:
    """
    Main configuration class that holds all components needed for the semantic search API.

    Attributes:
        openai_async_client (AsyncOpenAI): OpenAI client for API calls
        pg_db_async_engine (AsyncEngine): PostgreSQL database engine
        embedding_model (str): Name of the embedding model
        embedding_table (EmbeddingBaseTable): Database table for storing embeddings
        embedding_function (Callable): Function to generate embeddings
        vector_size (int): Dimension of embedding vectors
        re_rank_model (str): Model used for re-ranking search results
        query_refinement_model (str): Model used for query improvement
    """

    openai_async_client: AsyncOpenAI
    pg_db_async_engine: AsyncEngine
    embedding_model: str
    embedding_table: EmbeddingBaseTable
    embedding_function: Callable[[str | list[str]], Awaitable[CreateEmbeddingResponse]]
    vector_size: int
    re_rank_model: str
    query_refinement_model: str


async def check_if_valid_oai_model(client: AsyncOpenAI, model: str):
    """
    Verifies if the provided model name is a valid OpenAI model.

    Args:
        client (AsyncOpenAI): Authenticated OpenAI client
        model (str): Model name to verify

    Raises:
        BadConfigValue: If the model name is not valid
    """
    try:
        await client.models.retrieve(model=model)
        logger.info(f"Model: {model} verified as valid OpenAI model.")
    except NotFoundError:
        logger.error(f"Model name: {model} is not a valid openai model")
        raise BadConfigValue(
            f"Application requires valid OpenAI models, {model} could not be verified as a valid OpenAI model."
        )


async def get_config() -> Config:
    """
    Gathers and validates all configuration settings, and initializes core components.

    Returns:
        Config: Fully initialized configuration object

    Raises:
        BadConfigValue: If any configuration value is invalid
    """
    oai_client = create_async_openai_client()

    await asyncio.gather(
        check_if_valid_oai_model(oai_client, EMBEDDING_MODEL),
        check_if_valid_oai_model(oai_client, RE_RANK_MODEL),
        check_if_valid_oai_model(oai_client, QUERY_REFINEMENT_MODEL),
    )

    if "embedding" not in EMBEDDING_MODEL:
        raise BadConfigValue(
            f"Application expects an embedding model, {EMBEDDING_MODEL} does not contain the word embedding"
        )

    return Config(
        openai_async_client=oai_client,
        pg_db_async_engine=create_async_db_engine(),
        embedding_model=EMBEDDING_MODEL,
        embedding_table=create_embedding_table(),
        embedding_function=create_embedding_function(
            oai_client=oai_client,
            embedding_model=EMBEDDING_MODEL,
            vector_size=VECTOR_SIZE,
        ),
        vector_size=VECTOR_SIZE,
        re_rank_model=RE_RANK_MODEL,
        query_refinement_model=QUERY_REFINEMENT_MODEL,
    )
