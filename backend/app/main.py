"""FastAPI application entry point, lifecycle management, CORS, and routing."""

import logging
import os

# Prefer local cached models to avoid DNS/timeout delays when models are already cached
if not os.environ.get("HF_HUB_ONLINE"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import (
    analytics,
    documents,
    evaluation,
    experiments,
    qa,
    reports,
    retrieval,
    settings as settings_api,
)
from app.config import get_settings
from app.core.embeddings import get_embedding_manager
from app.core.vector_store import get_vector_store
from app.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_hallucination_backend")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handling startup initialization and teardown."""
    logger.info("Initializing %s...", settings.APP_NAME)

    # 1. Ensure required data directories exist
    for dir_path in [
        settings.FAISS_INDEX_PATH,
        settings.UPLOADS_PATH,
        settings.EXPERIMENTS_PATH,
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # 2. Initialize database schema
    try:
        init_db()
        logger.info("Database initialized successfully at %s", settings.DATABASE_URL)
    except Exception as e:
        logger.error("Failed to initialize database: %s", str(e))

    # 3. Load persisted FAISS index if exists
    vs = get_vector_store()
    try:
        loaded = vs.load_index()
        if loaded:
            logger.info("Loaded persisted FAISS index with %d chunks.", vs.get_stats().get("total_chunks", 0))
        else:
            logger.info("No prior FAISS index found. Ready for new documents.")
    except Exception as e:
        logger.warning("Could not load FAISS index: %s", str(e))

    # 4. Pre-load embedding model onto GPU/CPU
    em = get_embedding_manager()
    try:
        logger.info("Pre-loading embedding model '%s' on %s...", settings.EMBEDDING_MODEL, em.device)
        em.load_model()
        logger.info("Embedding model ready.")
    except Exception as e:
        logger.warning("Embedding model pre-load deferred: %s", str(e))

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for Evaluating and Mitigating RAG Hallucinations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
origins = settings.CORS_ORIGINS if settings.CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and format a clean JSON error response."""
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url.path),
        },
    )


# Include API Routers under /api prefix
api_prefix = "/api"
app.include_router(documents.router, prefix=api_prefix)
app.include_router(retrieval.router, prefix=api_prefix)
app.include_router(qa.router, prefix=api_prefix)
app.include_router(evaluation.router, prefix=api_prefix)
app.include_router(experiments.router, prefix=api_prefix)
app.include_router(analytics.router, prefix=api_prefix)
app.include_router(reports.router, prefix=api_prefix)
app.include_router(settings_api.router, prefix=api_prefix)

# Mount static uploads directory if exists
uploads_path = Path(settings.UPLOADS_PATH)
uploads_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health and component status check."""
    vs = get_vector_store()
    stats = vs.get_stats()
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": "1.0.0",
        "environment": "debug" if settings.DEBUG else "production",
        "vector_store": {
            "status": "online",
            "indexed_chunks": stats.get("total_chunks", 0),
            "indexed_documents": stats.get("total_documents", 0),
        },
    }


@app.get("/", tags=["Root"])
async def root_info():
    """Root info page directing to interactive API documentation."""
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "api_endpoints": {
            "documents": f"{api_prefix}/documents",
            "retrieval": f"{api_prefix}/retrieval",
            "qa": f"{api_prefix}/qa",
            "evaluation": f"{api_prefix}/evaluation",
            "experiments": f"{api_prefix}/experiments",
            "analytics": f"{api_prefix}/analytics",
            "reports": f"{api_prefix}/reports",
            "settings": f"{api_prefix}/settings",
        },
    }
