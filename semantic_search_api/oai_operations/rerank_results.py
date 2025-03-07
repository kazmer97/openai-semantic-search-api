import json
import logging
from openai import AsyncOpenAI
from pydantic import BaseModel
from semantic_search_api import SimilaritySearchResult

logger = logging.getLogger(__name__)


class ResponseFormat(BaseModel):
    """Pydantic model for the expected response format from the OpenAI reranking.

    Attributes:
        updated_rankings (list[int]): List of new rankings for the search results
    """

    updated_rankings: list[int]


async def re_rank_results(
    oai_client: AsyncOpenAI,
    input_list: list[SimilaritySearchResult],
    user_query: str,
    enhanced_user_query: str,
    additional_user_context: dict | None = None,
    model="gpt-4o-2024-11-20",
) -> list[SimilaritySearchResult]:
    """Re-rank search results using OpenAI's language model to improve relevance.

    Args:
        oai_client (AsyncOpenAI): OpenAI client instance for making API calls
        input_list (list[SimilaritySearchResult]): List of initial search results with rankings
        user_query (str): Original query from the user
        enhanced_user_query (str): Processed/enhanced version of the user query
        additional_user_context (dict | None, optional): Additional context to help with ranking. Defaults to None.
        model (str, optional): OpenAI model to use. Defaults to "gpt-4o-2024-11-20".

    Returns:
        list[SimilaritySearchResult]: Re-ranked search results, maintaining original list if reranking fails validation
    """
    if not additional_user_context:
        additional_user_context = {}
    result = await oai_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "developer",
                "content": "think of yourself as a very sophisticated shopping assistant. I will give you a list of products, in a json format. "
                "They will contain a l2_distance similarity score from a vector search, a json formatted data field with everything we know about the product, and ranking column."
                "Looking at this I want you to produce a JSON output that re-ranks the relevance of these items based on the user query."
                "1. Is the highest ranking, if there is a 0 then 0 becomes the highest ranking."
                "I want you to create a list where you place the existing rankings in the order that you think would be better."
                "\nMake sure to include one ranking for each document in the response.!!!",
            },
            {
                "role": "user",
                "content": f"<user_query_string>{user_query}<\\user_query_string>\n"
                f"<enhanced_user_query_string>{enhanced_user_query}<\\enhanced_user_query_string>\n"
                f"<additional_user_context>{additional_user_context}<\\additional_user_context>\n"
                f"<vector_search_product_list>{json.dumps(input_list)}<\\vector_search_product_list>\n",
            },
        ],
        response_format=ResponseFormat,
    )

    reranking = result.choices[0].message.parsed

    original_ranking = [rec["ranking"] for rec in input_list]

    updated_ranking_list = reranking.updated_rankings

    if (
        not updated_ranking_list
        or len(updated_ranking_list) != len(set(updated_ranking_list))
        or len(updated_ranking_list) != len(original_ranking)
        or (max(original_ranking) != max(updated_ranking_list))
        or (min(original_ranking) != min(updated_ranking_list))
    ):
        logger.warning(
            "Re-ranking output is not reliable enough and will not be applied.",
            f"original rankings: {original_ranking}\n"
            f"new ranking list: {updated_ranking_list}",
        )
        return input_list

    updated_list = input_list.copy()

    for ranking, rec in zip(updated_ranking_list, updated_list):
        rec["ranking"] = ranking

    return updated_list
