from typing import Annotated
from fastapi import APIRouter, Body, Request
from pydantic import BaseModel


from semantic_search_api.oai_operations.search_workflow import (
    get_semantic_search_results,
)

router = APIRouter(tags=["semantic_search"])


class QueryInputModel(BaseModel):
    query: str
    additional_user_context: dict
    use_re_rank: bool = False
    number_of_results: int = 10


class ResultObject(BaseModel):
    data: dict
    ranking: int
    similarity: float


class QueryOutput(BaseModel):
    query_results: list[ResultObject]


@router.post("/semantic_query", response_model=QueryOutput)
async def semantic_search(
    input_body: Annotated[QueryInputModel, Body], request: Request
) -> QueryOutput:
    """Execute a semantic search query using the provided input parameters.

    Args:
        input_body (QueryInputModel): The search query parameters including the query text,
            additional context, re-ranking preferences, and number of desired results.
        request (Request): The FastAPI request object containing application state and configuration.

    Returns:
        QueryOutput: A container with a list of search results, each containing the matched data,
            ranking position, and similarity score.
    """
    result = await get_semantic_search_results(
        oai_client=request.app.state.app_config.openai_async_client,
        database_engine=request.app.state.app_config.pg_db_async_engine,
        text_question=input_body.query,
        additional_user_context=input_body.additional_user_context,
        embedding_function=request.app.state.app_config.embedding_function,
        embedding_table=request.app.state.app_config.embedding_table,
        use_re_rank=input_body.use_re_rank,
        limit=input_body.number_of_results,
        re_rank_model=request.app.state.app_config.re_rank_model,
        query_refinement_model=request.app.state.app_config.query_refinement_model,
    )

    return QueryOutput(query_results=result)
