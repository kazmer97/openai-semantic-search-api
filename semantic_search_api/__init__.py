from typing import TypedDict


class SimilaritySearchResult(TypedDict):
    """Represents a result from a semantic similarity search.

    A typed dictionary that contains search result metadata including the actual data,
    its ranking position, and similarity score.
    """

    data: dict  # The content/payload of the search result
    ranking: int  # The position of this result in the ranked results list
    similarity: float  # The calculated similarity score (0-1) for this result
