import json
import logging
from typing import Awaitable, Callable
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncEngine

from semantic_search_api import SimilaritySearchResult
from semantic_search_api.database.database_query import search_similar_documents
from semantic_search_api.database.tables import EmbeddingBaseTable
from semantic_search_api.oai_operations.rerank_results import re_rank_results
from openai.types.create_embedding_response import CreateEmbeddingResponse

logger = logging.getLogger(__name__)


async def _refine_semantic_search_query(
    oai_client: AsyncOpenAI,
    text_question: str,
    additional_user_context: dict,
    model: str = "gpt-4o-2024-11-20",
) -> str:
    """Refine a user's search query using GPT model and additional context.

    Args:
        oai_client (AsyncOpenAI): OpenAI client instance
        text_question (str): Original search query from user
        additional_user_context (dict): Additional context about the user/search
        model (str, optional): GPT model to use. Defaults to "gpt-4o-2024-11-20"

    Returns:
        str: Refined and improved search query
    """
    response = await oai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "developer",
                "content": "you are the core part of a semantic search system, your function is to return a single line string that improves the users query using the user context information.",
            },
            {
                "role": "user",
                "content": f"<additional_user_context>\n{json.dumps(additional_user_context)}\n<\\additional_user_context>",
            },
            {"role": "user", "content": text_question},
        ],
    )
    improved_query_string = response.choices[0].message.content
    logging.info(f"Improved user query: {improved_query_string}")
    return improved_query_string


async def get_semantic_search_results(
    oai_client: AsyncOpenAI,
    text_question: str,
    database_engine: AsyncEngine,
    embedding_function: Callable[[str | list[str]], Awaitable[CreateEmbeddingResponse]],
    embedding_table: EmbeddingBaseTable,
    additional_user_context: dict | None = None,
    use_re_rank: bool = False,
    query_refinement_model="gpt-4o-2024-11-20",
    re_rank_model="gpt-4o-2024-11-20",
    limit: int = 10,
) -> list[SimilaritySearchResult]:
    """Perform semantic search with query refinement and optional re-ranking.

    Args:
        oai_client (AsyncOpenAI): OpenAI client instance
        text_question (str): Original search query from user
        database_engine (AsyncEngine): SQLAlchemy async database engine
        embedding_function (Callable): Function to generate embeddings
        embedding_table (EmbeddingBaseTable): Database table containing embeddings
        additional_user_context (dict, optional): Additional context about the user/search. Defaults to None
        use_re_rank (bool, optional): Whether to re-rank results. Defaults to False
        query_refinement_model (str, optional): Model for query refinement. Defaults to "gpt-4o-2024-11-20"
        re_rank_model (str, optional): Model for re-ranking. Defaults to "gpt-4o-2024-11-20"
        limit (int, optional): Maximum number of results. Defaults to 10

    Returns:
        list[SimilaritySearchResult]: List of search results ordered by relevance
    """
    if not additional_user_context:
        additional_user_context = {}

    search_query = await _refine_semantic_search_query(
        oai_client=oai_client,
        text_question=text_question,
        additional_user_context=additional_user_context,
        model=query_refinement_model,
    )

    embedding_reponse = await embedding_function(input=search_query)

    query_vector = embedding_reponse.data[0].embedding

    sim_search_results = await search_similar_documents(
        engine=database_engine,
        query_embedding=query_vector,
        limit=limit,
        table=embedding_table,
    )

    if not use_re_rank:
        return sim_search_results

    re_ranking_results = await re_rank_results(
        oai_client=oai_client,
        model=re_rank_model,
        input_list=sim_search_results,
        user_query=text_question,
        enhanced_user_query=search_query,
        additional_user_context=additional_user_context,
    )

    return re_ranking_results


if __name__ == "__main__":
    import asyncio

    # Example usage in a real script:
    from semantic_search_api.config import get_config

    config = get_config()
    asyncio.run(
        get_semantic_search_results(
            database_engine=config.pg_db_async_engine,
            oai_client=config.openai_async_client,
            text_question="looking for an earring for my girlfriends 30th birthday",
            additional_user_context={"gender": "male", "age": "35"},
        )
    )
