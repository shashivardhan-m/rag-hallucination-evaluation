"""API endpoints for inspecting and updating runtime system settings."""

from fastapi import APIRouter
from app.config import get_settings
from app.models.schemas import SettingsRequest, SettingsResponse

router = APIRouter(prefix="/settings", tags=["Settings"])
settings = get_settings()


@router.get("", response_model=SettingsResponse, include_in_schema=False)
@router.get("/", response_model=SettingsResponse)
async def get_current_settings():
    """Retrieve active runtime configuration."""
    return SettingsResponse(
        app_name=settings.APP_NAME,
        debug=settings.DEBUG,
        default_llm_backend=settings.DEFAULT_LLM_BACKEND,
        default_groq_model=settings.DEFAULT_GROQ_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        reranker_model=settings.RERANKER_MODEL,
        default_top_k=settings.DEFAULT_TOP_K,
        default_chunk_size=settings.DEFAULT_CHUNK_SIZE,
        default_chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
        max_file_size_mb=settings.MAX_FILE_SIZE_MB,
        groq_key_set=bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip()),
        openai_key_set=bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip()),
        ollama_base_url=settings.OLLAMA_BASE_URL,
        faiss_index_path=settings.FAISS_INDEX_PATH,
        database_url=settings.DATABASE_URL,
    )


@router.patch("", response_model=SettingsResponse, include_in_schema=False)
@router.patch("/", response_model=SettingsResponse)
async def update_settings(payload: SettingsRequest):
    """Dynamically update runtime settings in memory."""
    if payload.debug is not None:
        settings.DEBUG = payload.debug
    backend_val = payload.default_llm_backend or payload.llm_backend
    if backend_val is not None:
        settings.DEFAULT_LLM_BACKEND = backend_val
    model_val = payload.default_groq_model or payload.llm_model
    if model_val is not None:
        settings.DEFAULT_GROQ_MODEL = model_val
    if payload.embedding_model is not None:
        settings.EMBEDDING_MODEL = payload.embedding_model
    if payload.reranker_model is not None:
        settings.RERANKER_MODEL = payload.reranker_model
    top_k_val = payload.default_top_k or payload.top_k
    if top_k_val is not None:
        settings.DEFAULT_TOP_K = top_k_val
    chunk_size_val = payload.default_chunk_size or payload.chunk_size
    if chunk_size_val is not None:
        settings.DEFAULT_CHUNK_SIZE = chunk_size_val
    overlap_val = payload.default_chunk_overlap or payload.chunk_overlap
    if overlap_val is not None:
        settings.DEFAULT_CHUNK_OVERLAP = overlap_val
    if payload.groq_api_key is not None:
        settings.GROQ_API_KEY = payload.groq_api_key
    if payload.openai_api_key is not None:
        settings.OPENAI_API_KEY = payload.openai_api_key
    if payload.ollama_base_url is not None:
        settings.OLLAMA_BASE_URL = payload.ollama_base_url

    return await get_current_settings()
