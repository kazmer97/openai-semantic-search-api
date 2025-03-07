import pytest


@pytest.mark.asyncio(loop_scope="module")
async def test_check_if_query_returns_expected_data(
    create_test_db, load_data_into_test_db
):
    """Test the semantic search functionality with a specific query for men's socks.

    This test verifies that when searching for boyfriend's socks with a female user context,
    the search results include appropriate men's socks products.

    Args:
        create_test_db: Fixture that creates a test database
        load_data_into_test_db: Fixture that loads test data and returns config

    Returns:
        None

    Raises:
        AssertionError: If the expected men's socks product is not found in results
    """
    from semantic_search_api.oai_operations.search_workflow import (
        get_semantic_search_results,
    )

    config, _ = load_data_into_test_db
    query_text = "socks for my boyfriend"

    results = await get_semantic_search_results(
        additional_user_context={
            "gender": "female",
        },
        database_engine=config.pg_db_async_engine,
        embedding_table=config.embedding_table,
        embedding_function=config.embedding_function,
        limit=5,
        oai_client=config.openai_async_client,
        use_re_rank=True,
        text_question=query_text,
    )

    assert (
        any(
            "YUEDGE 5 Pairs Men's Moisture Control Cushioned Dry Fit Casual Athletic Crew Socks for Men (Blue, Size 9-12)"
            in v["data"]["title"]
            for v in results
        ),
        "The men's socks should be returned as one of the results for this prompt on the test dataset",
    )
