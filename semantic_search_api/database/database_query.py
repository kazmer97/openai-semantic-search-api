import json
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from semantic_search_api.database.tables import EmbeddingBaseTable
from semantic_search_api import SimilaritySearchResult

logger = logging.getLogger(__name__)


async def search_similar_documents(
    engine: AsyncEngine,
    table: EmbeddingBaseTable,
    query_embedding: list[float],
    limit: int = 5,
) -> list[SimilaritySearchResult]:
    """Searches for documents with similar vector embeddings using L2 distance.

    Executes an asynchronous database query to find documents whose embeddings
    are closest to the provided query embedding vector using L2 distance metric.
    Results are ordered by similarity (closest first).

    Args:
        engine: SQLAlchemy async engine for database connections.
        table: SQLAlchemy table model containing the embeddings.
        query_embedding: List of floats representing the embedding vector to search against.
        limit: Maximum number of results to return. Defaults to 5.

    Returns:
        List of SimilaritySearchResult objects, each containing:
            - data: The original document data as a dictionary
            - similarity: L2 distance score (lower is more similar)
            - ranking: Position in results (1-based)

    Raises:
        SQLAlchemyError: If database query fails.
        json.JSONDecodeError: If stored document data is not valid JSON.
    """

    async with AsyncSession(engine) as session:
        similarity_expr = table.embedding.l2_distance(query_embedding).label(
            "similarity"
        )

        stmt = (
            select(table.json_content, similarity_expr)
            .order_by(similarity_expr)
            .limit(limit)
        )

        result = await session.execute(
            stmt,
        )

        similar_docs = result.all()

        return [
            SimilaritySearchResult(
                data=json.loads(row.json_content),
                similarity=float(row.similarity),
                ranking=i + 1,
            )
            for i, row in enumerate(similar_docs)
        ]
