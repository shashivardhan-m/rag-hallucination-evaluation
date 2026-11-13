"""Text processing and formatting utilities."""

import re
from typing import Any, List, Union


def clean_text(text: str) -> str:
    """Remove control characters, excessive whitespace, and normalize line breaks."""
    if not text:
        return ""
    # Strip non-printable/control chars except \n, \r, \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Replace non-breaking spaces and irregular spaces with single space
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace 3 or more consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple horizontal spaces or tabs with single space
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Re-join lines
    cleaned = "\n".join(lines).strip()
    return cleaned


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for English text (~4 characters per token)."""
    if not text:
        return 0
    # Common rule of thumb: max(1, len(text) // 4) or words * 1.33
    words = len(text.split())
    by_chars = max(1, len(text) // 4)
    by_words = int(words * 1.33)
    return max(by_chars, by_words)


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars with an ellipsis if exceeded."""
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars].rstrip() + "..."


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences respecting abbreviations, decimal numbers, and punctuation."""
    if not text:
        return []
    # Regex splitting on punctuation (. ! ?) followed by whitespace and capital letter or end of string
    pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])"
    parts = re.split(pattern, text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text.strip()]


def format_context_for_prompt(chunks: List[Union[dict, Any]]) -> str:
    """Format retrieved context chunks into clear markdown sections for LLM prompting."""
    if not chunks:
        return "No relevant context found."

    formatted_sections = []
    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            source_file = chunk.get("filename", "Unknown Document")
            chunk_idx = chunk.get("chunk_index", idx)
            content = chunk.get("content", "").strip()
            score = chunk.get("score")
        else:
            source_file = getattr(chunk, "filename", "Unknown Document")
            chunk_idx = getattr(chunk, "chunk_index", idx)
            content = getattr(chunk, "content", "").strip()
            score = getattr(chunk, "score", None)

        score_info = f" (relevance: {score:.3f})" if score is not None else ""
        header = f"[Context Source {idx}: {source_file} (Chunk {chunk_idx}){score_info}]"
        formatted_sections.append(f"{header}\n{content}")

    return "\n\n".join(formatted_sections)
