import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


def test_row_to_string():
    """Test the create_string_from_row function.

    Verifies that the function correctly converts a dictionary row into a
    formatted string with key-value pairs.
    """
    from semantic_search_api.database.data_ingestion import create_string_from_row

    test_row = {"title": "Ilyad", "writer": "Homer"}
    test = create_string_from_row(row=test_row)

    print(test)
    assert test == "title: Ilyad writer: Homer"


@pytest.mark.asyncio(loop_scope="module")
async def test_check_uploaded_data(load_data_into_test_db):
    """Test the data upload process to the database.

    Verifies that the number of rows in the database matches the number
    of lines in the test data file.

    Args:
        load_data_into_test_db: Fixture that loads test data into database
    """
    config, _ = load_data_into_test_db
    engine = config.pg_db_async_engine
    test_table = config.embedding_table

    async with AsyncSession(engine) as session:
        result = await session.execute(select(func.count()).select_from(test_table))

        count = result.scalar()
        print("Row count:", count)

    with open("tests/small_test_data.jsonl", "r", encoding="utf-8") as f:
        line_count = sum(1 for line in f if line.strip())

    assert line_count == count
