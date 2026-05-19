import fitz  # PyMuPDF
from dataclasses import dataclass

CHUNK_SIZE = 600   # ký tự mỗi chunk
CHUNK_OVERLAP = 80


@dataclass
class Chunk:
    content: str
    page_num: int


def extract_chunks(file_bytes: bytes, filename: str) -> tuple[list[Chunk], int]:
    """Extract text from PDF/DOCX bytes and split into overlapping chunks."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        pages = _extract_pdf(file_bytes)
    else:
        # Fallback: treat as plain text
        text = file_bytes.decode("utf-8", errors="ignore")
        pages = [(text, 1)]

    chunks: list[Chunk] = []
    for text, page_num in pages:
        text = text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(content=chunk_text, page_num=page_num))
            start += CHUNK_SIZE - CHUNK_OVERLAP

    total_pages = max((p for _, p in pages), default=0)
    return chunks, total_pages


def _extract_pdf(file_bytes: bytes) -> list[tuple[str, int]]:
    pages: list[tuple[str, int]] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append((text, i))
    return pages
