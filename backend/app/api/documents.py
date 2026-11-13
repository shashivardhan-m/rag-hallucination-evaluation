"""API endpoints for document upload, processing, chunk indexing, and management."""

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.document_processor import DocumentProcessor
from app.core.embeddings import get_embedding_manager
from app.core.vector_store import get_vector_store
from app.database import get_db
from app.models.db_models import Document, DocumentChunk
from app.models.schemas import (
    ChunkInfo,
    ChunkListResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/documents", tags=["Documents"])
settings = get_settings()
processor = DocumentProcessor()
embedding_manager = get_embedding_manager()
vector_store = get_vector_store()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF or TXT document, extract and chunk text, compute embeddings,
    and index in FAISS vector store and SQLite database.
    """
    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower().lstrip(".")

    if ext not in ["pdf", "txt", "md"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: '.{ext}'. Supported formats: PDF, TXT, MD",
        )

    # Ensure uploads directory exists
    uploads_dir = Path(settings.UPLOADS_PATH)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    saved_filename = f"{doc_id}_{filename}"
    saved_path = uploads_dir / saved_filename

    # Stream file to disk while enforcing max file size
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    total_bytes = 0

    try:
        with open(saved_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB",
                    )
                buffer.write(chunk)
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    # Process and chunk document in thread pool to prevent blocking event loop
    effective_chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
    effective_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP

    try:
        processed = await asyncio.to_thread(
            processor.process_document,
            file_path=str(saved_path),
            filename=filename,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_overlap,
        )
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error extracting text from document: {str(e)}",
        )

    chunks = processed.get("chunks", [])
    if not chunks:
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid text chunks could be extracted from this document.",
        )

    # Generate embeddings
    chunk_texts = [c["content"] for c in chunks]
    try:
        embeddings = await asyncio.to_thread(
            embedding_manager.get_embeddings,
            texts=chunk_texts,
            batch_size=32,
        )
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}",
        )

    # Assign IDs and metadata to chunks
    for c in chunks:
        c["chunk_id"] = str(uuid.uuid4())
        c["document_id"] = doc_id
        c["filename"] = filename
        c["embedding_model"] = embedding_manager.current_model_name or settings.EMBEDDING_MODEL

    # Add to FAISS and persist
    await asyncio.to_thread(vector_store.add_chunks, chunks, embeddings, doc_id)
    await asyncio.to_thread(vector_store.save_index)

    # Save Document and Chunks to SQLite DB
    meta_json = json.dumps({
        "saved_path": str(saved_path),
        "total_chars": processed.get("total_chars", 0),
        "total_tokens_estimate": processed.get("total_tokens_estimate", 0),
        "chunk_size": effective_chunk_size,
        "chunk_overlap": effective_overlap,
    })

    db_doc = Document(
        id=doc_id,
        filename=filename,
        file_type=ext,
        file_size=total_bytes,
        num_chunks=len(chunks),
        description=description,
        metadata_json=meta_json,
    )
    db.add(db_doc)

    for c in chunks:
        db_chunk = DocumentChunk(
            id=c["chunk_id"],
            document_id=doc_id,
            chunk_index=c["chunk_index"],
            content=c["content"],
            embedding_model=c["embedding_model"],
            char_count=c["char_count"],
            token_estimate=c["token_estimate"],
        )
        db.add(db_chunk)

    db.commit()
    db.refresh(db_doc)

    return DocumentUploadResponse(
        id=db_doc.id,
        filename=db_doc.filename,
        file_type=db_doc.file_type,
        file_size=db_doc.file_size,
        num_chunks=db_doc.num_chunks,
        upload_time=db_doc.upload_time,
        message=f"Document processed and {len(chunks)} chunks indexed into vector store.",
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Retrieve all indexed documents with pagination."""
    query = db.query(Document).order_by(Document.upload_time.desc())
    total = query.count()
    docs = query.offset(skip).limit(limit).all()

    return DocumentListResponse(
        total=total,
        documents=[DocumentInfo.model_validate(d) for d in docs],
    )


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Retrieve details of a single document."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {doc_id} not found",
        )
    return DocumentInfo.model_validate(doc)


@router.get("/{doc_id}/chunks", response_model=ChunkListResponse)
async def get_document_chunks(
    doc_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Retrieve all chunks belonging to a document."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {doc_id} not found",
        )

    query = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).order_by(DocumentChunk.chunk_index.asc())
    total = query.count()
    chunks = query.offset(skip).limit(limit).all()

    return ChunkListResponse(
        total=total,
        chunks=[ChunkInfo.model_validate(c) for c in chunks],
    )


@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Delete document and its chunks from the database and FAISS vector index."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {doc_id} not found",
        )

    # Remove stored physical file if metadata contains it
    try:
        meta = json.loads(doc.metadata_json or "{}")
        file_path = meta.get("saved_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

    # Remove from FAISS
    await asyncio.to_thread(vector_store.delete_document, doc_id)
    await asyncio.to_thread(vector_store.save_index)

    # Delete from DB
    db.delete(doc)
    db.commit()

    return {"message": f"Document {doc_id} and all related chunks deleted successfully"}
