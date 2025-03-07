"""
Main FastAPI application module for the Semantic Search API.

This module initializes the FastAPI application with CORS middleware
and includes the main API router. It also handles application lifecycle
through the lifespan context manager.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from semantic_search_api.api.main import api_router
from semantic_search_api.config import get_config
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.

    Handles initialization and cleanup of application resources.
    Currently sets up the application configuration on startup.

    Args:
        app (FastAPI): The FastAPI application instance
    """
    app.state.app_config = await get_config()
    yield


app = FastAPI(
    title="OpenAI home challenge SemanticSearch API",
    openapi_url=f"/api/v1/openapi.json",
    lifespan=lifespan,
)


app.include_router(api_router, prefix="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
