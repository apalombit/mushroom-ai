"""
Mushroom Lookalikes Finder — FastAPI application.

Endpoints:
    POST /api/v1/lookalikes         → Find lookalikes for a species
    GET  /api/v1/species/{name}     → Get species profile
    GET  /api/v1/species            → List all species in database
    GET  /health                    → Health check
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from db.connection import check_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify database connectivity
    if not check_connection():
        raise RuntimeError("Cannot connect to PostgreSQL. Is docker compose up?")
    yield


app = FastAPI(
    title="Mushroom Lookalikes Finder",
    description="Find dangerous lookalike species using feature-based similarity + LLM explanations",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "mushroom-lookalikes"}
