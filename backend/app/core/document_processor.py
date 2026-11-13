"""Document text extraction, cleaning, and sentence-aware chunking."""

import os
from pathlib import Path
from typing import Any, Dict, List
import pymupdf as fitz  # PyMuPDF (modern import, avoids FutureWarning)

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
        """Extract text from a PDF file using PyMuPDF (fitz)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        extracted_text_parts: List[str] = []
        try:
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    if text and text.strip():
                        extracted_text_parts.append(text)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF file {file_path}: {e}") from e

        return "\n\n".join(extracted_text_parts)

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract raw text from a TXT or Markdown file with fallback encodings."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found at: {file_path}")

        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # If all fail, read with replace error handler
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ) -> List[Dict[str, Any]]:
        """
        Split text into character-sized chunks respecting sentence boundaries.
        Each dictionary contains: chunk_index, content, char_count, token_estimate.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 4)

        sentences = split_into_sentences(cleaned)
        if not sentences:
            return []

        # If individual sentence is longer than chunk_size, split by words
        normalized_sentences: List[str] = []
        for s in sentences:
            if len(s) > chunk_size:
                words = s.split()
                cur_seg: List[str] = []
                cur_len = 0
                for w in words:
                    w_len = len(w) + 1
                    if cur_len + w_len > chunk_size and cur_seg:
                        normalized_sentences.append(" ".join(cur_seg))
                        cur_seg = [w]
                        cur_len = len(w)
                    else:
                        cur_seg.append(w)
                        cur_len += w_len
                if cur_seg:
                    normalized_sentences.append(" ".join(cur_seg))
            else:
                normalized_sentences.append(s)

        chunks: List[Dict[str, Any]] = []
        current_chunk_sentences: List[str] = []
        current_length = 0
        chunk_index = 0

        for sentence in normalized_sentences:
            sentence_len = len(sentence) + (1 if current_chunk_sentences else 0)

            # If adding this sentence exceeds chunk_size and we already have content
            if current_length + sentence_len > chunk_size and current_chunk_sentences:
                chunk_str = " ".join(current_chunk_sentences).strip()
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": chunk_str,
                    "char_count": len(chunk_str),
                    "token_estimate": estimate_tokens(chunk_str),
                })
                chunk_index += 1

                # Build overlap from the tail of current_chunk_sentences
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) + 1 <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_chunk_sentences = list(overlap_sentences)
                current_length = sum(len(s) for s in current_chunk_sentences) + max(
                    0, len(current_chunk_sentences) - 1
                )

            current_chunk_sentences.append(sentence)
            current_length += sentence_len

        # Append remaining sentences as the final chunk
        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences).strip()
            # If the final chunk is identical to the previous chunk due to overlap, avoid duplicate
            if not chunks or chunks[-1]["content"] != chunk_str:
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": chunk_str,
                    "char_count": len(chunk_str),
                    "token_estimate": estimate_tokens(chunk_str),
                })

        return chunks

    def process_document(
        self,
        file_path: str,
        filename: str,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ) -> Dict[str, Any]:
        """
        Extract text, clean, chunk, and return full metadata structure.
        """
        path = Path(file_path)
        ext = path.suffix.lower().lstrip(".")
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        if ext == "pdf":
            raw_text = self.extract_text_from_pdf(file_path)
        elif ext in ["txt", "md", "csv", "json"]:
            raw_text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: .{ext}. Supported formats: PDF, TXT, MD")

        cleaned_text = self.clean_text(raw_text)
        if not cleaned_text:
            raise ValueError(f"Document {filename} contains no extractable text.")

        chunks = self.split_into_chunks(
            cleaned_text,
            chunk_size=chunk_size or self.default_chunk_size,
            chunk_overlap=chunk_overlap or self.default_chunk_overlap,
        )

        return {
            "filename": filename,
            "file_type": ext,
            "file_size": file_size,
            "num_chunks": len(chunks),
            "chunks": chunks,
            "total_chars": sum(c["char_count"] for c in chunks),
            "total_tokens_estimate": sum(c["token_estimate"] for c in chunks),
        }
