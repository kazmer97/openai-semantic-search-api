"""
Module for handling data loading operations in the semantic search API.
Provides endpoints and utilities for loading data into the vector database.
"""

from typing import Annotated
from fastapi import APIRouter, Body, Request
from fastapi.exceptions import HTTPException
from http import HTTPStatus
from pydantic import BaseModel, Field
from pathlib import Path


from semantic_search_api.database.data_ingestion import store_embeddings_into_vector_db

router = APIRouter(tags=["data_load"])


class DataloadInputModel(BaseModel):
    """
    Input model for data loading operations.

    Attributes:
        data_path (str): Path to the JSONL file to be uploaded
        cutoff (int, optional): Maximum number of records to process
        batch_size (int): Number of records to process in each batch
        concurrency (int): Maximum number of concurrent API and database calls
    """

    data_path: str = Field(
        ..., description="the name of the jsonl file you'd like uploaded"
    )
    cutoff: int = Field(
        None,
        description="The number of records you'd like to process from the file, by default the whole file will be processed.",
    )
    batch_size: int = Field(
        10, description="Number of records to be batched together for processing", le=15
    )
    concurrency: int = Field(
        5,
        description="Number of concurrent calls allowed to the openai api and the database during ingestion",
        le=100,
    )


class DataloadOutput(BaseModel):
    """
    Output model for data loading operations.

    Attributes:
        number_of_records_processed (int): Total number of records successfully processed
    """

    number_of_records_processed: int


def get_data_path(filename: str) -> Path:
    """
    Resolves the full path to a data file, accounting for Docker environment.

    Args:
        filename (str): Name of the file to locate

    Returns:
        Path: Full path to the data file
    """
    data_dir = Path("/data")
    if data_dir.exists() and data_dir.is_dir():
        return data_dir / filename
    return Path(filename)


@router.post("/data_load", response_model=DataloadOutput)
async def semantic_search(
    input_body: Annotated[DataloadInputModel, Body], request: Request
):
    """
    Endpoint for loading and processing data into the vector database.

    Args:
        input_body (DataloadInputModel): Input parameters for data loading
        request (Request): FastAPI request object containing application state

    Returns:
        DataloadOutput: Number of records processed

    Raises:
        HTTPException: If the specified data file is not found
    """
    data_path = get_data_path(input_body.data_path)

    if not data_path.is_file():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Data file not found at {data_path}",
        )

    processed = await store_embeddings_into_vector_db(
        engine=request.app.state.app_config.pg_db_async_engine,
        data_path=data_path,
        embedding_function=request.app.state.app_config.embedding_function,
        cutoff=input_body.cutoff,
        batch_size=input_body.batch_size,
        concurrency=input_body.concurrency,
        table=request.app.state.app_config.embedding_table,
    )

    return DataloadOutput(number_of_records_processed=processed)
