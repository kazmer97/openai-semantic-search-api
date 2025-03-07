"""
Main API router configuration module.

This module sets up the main FastAPI router and includes sub-routers
for query and data loading endpoints. It serves as the central routing
configuration for the semantic search API.
"""

from fastapi import APIRouter

from semantic_search_api.api.routes import query, data_load

api_router = APIRouter()
api_router.include_router(query.router)
api_router.include_router(data_load.router)
