"""Document text extraction, cleaning, and sentence-aware chunking."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # older pymupdf alias
    except ImportError:
        fitz = None

try:
    import pypdf
except ImportError:
    pypdf = None

from app.utils.text_utils import clean_text, estimate_tokens, split_into_sentences


class DocumentProcessor:
    """Handles text extraction from PDF/TXT files and chunking."""

    def __init__(self, default_chunk_size: int = 512, default_chunk_overlap: int = 128) -> None:
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    def clean_text(self, text: str) -> str:
        """Sanitize text by removing non-printable characters and normalizing spaces."""
        return clean_text(text)

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF (fitz) or pypdf fallback."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        extracted_text_parts: List[str] = []

        if fitz is not None:
            try:
                with fitz.open(file_path) as doc:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text = page.get_text("text")
                        if text and text.strip():
                            extracted_text_parts.append(text)
                return "\n\n".join(extracted_text_parts)
            except Exception as e:
                # Fall through to pypdf
                pass

        if pypdf is not None:
            try:
                reader = pypdf.PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        extracted_text_parts.append(text)
                return "\n\n".join(extracted_text_parts)
            except Exception as e:
                raise RuntimeError(f"Failed to read PDF file {file_path} with pypdf: {e}") from e

        # If neither PDF library is available
        raise RuntimeError("No PDF extraction library available (PyMuPDF or pypdf). Please install PyMuPDF.")

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from a plain UTF-8 text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found at: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read TXT file {file_path}: {e}") from e

    def split_into_chunks(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment text into sliding-window overlapping chunks with sentence boundary awareness.
        """
        size = chunk_size or self.default_chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else self.default_chunk_overlap

        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        sentences = split_into_sentences(cleaned)
        if not sentences:
            return []

        chunks: List[Dict[str, Any]] = []
        current_chunk_sentences: List[str] = []
        current_char_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # include space

            if current_char_count + sentence_len > size and current_chunk_sentences:
                # Flush current chunk
                chunk_text = " ".join(current_chunk_sentences).strip()
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "char_count": len(chunk_text),
                    "token_estimate": estimate_tokens(chunk_text),
                })
                chunk_index += 1

                # Retain overlap sentences for sliding window
                overlap_sentences: List[str] = []
                overlap_chars = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_chars + len(s) + 1 <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_chars += len(s) + 1
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_char_count = overlap_chars

            current_chunk_sentences.append(sentence)
            current_char_count += sentence_len

        # Flush any remaining text in the buffer
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            chunks.append({
                "chunk_index": chunk_index,
                "content": chunk_text,
                "char_count": len(chunk_text),
                "token_estimate": estimate_tokens(chunk_text),
            })

        return chunks

    def process_document(
        self,
        file_path: str,
        filename: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract text from file and partition into chunks."""
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            raw_text = self.extract_text_from_pdf(file_path)
            file_type = "pdf"
        elif ext in (".txt", ".md", ".json", ".log"):
            raw_text = self.extract_text_from_txt(file_path)
            file_type = "txt"
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        chunks = self.split_into_chunks(raw_text, chunk_size, chunk_overlap)
        total_tokens = sum(c["token_estimate"] for c in chunks)

        return {
            "filename": filename,
            "file_type": file_type,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else len(raw_text),
            "num_chunks": len(chunks),
            "total_chars": len(raw_text),
            "total_tokens_estimate": total_tokens,
            "raw_text": raw_text,
            "chunks": chunks,
        }
