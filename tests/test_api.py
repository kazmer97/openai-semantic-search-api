"""
API Testing Module for Semantic Search Service.

This module contains integration tests for the semantic search API endpoints,
verifying the search functionality, result relevance, and response format.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio(loop_scope="session")
async def test_check_if_api_returns_expected_data(replace_env_variables):
    """
    Test the semantic query API endpoint returns expected search results.

    Tests if the API correctly processes a query for men's socks with additional
    user context and returns relevant results including a specific product.
    The test verifies both the API response structure and the relevance of
    returned results for a specific use case.

    Args:
        replace_env_variables: Fixture to set up environment variables for testing

    Returns:
        None

    Raises:
        AssertionError: If the response status is not 200 or if the expected
            product is not found in the results
    """
    replace_env_variables
    from semantic_search_api.main import app

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/semantic_query",
            json={
                "query": "socks for my boyfriend",
                "additional_user_context": {"gender": "female", "age": "25-30"},
                "use_re_rank": True,
                "number_of_results": 5,
            },
        )

    assert response.status_code == 200

    assert (
        any(
            "YUEDGE 5 Pairs Men's Moisture Control Cushioned Dry Fit Casual Athletic Crew Socks for Men (Blue, Size 9-12)"
            in v["data"]["title"]
            for v in response.json()["query_results"]
        ),
        "The men's socks should be returned as one of the results for this prompt on the test dataset",
    )
