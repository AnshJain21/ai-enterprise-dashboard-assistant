"""Extract plain text from uploaded PDF / DOCX / TXT files, then chunk it."""
import io
from pypdf import PdfReader
from docx import Document


def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    if ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: .{ext}")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Simple sliding-window chunker on characters (good enough for reports)."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
