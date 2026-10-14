from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class Document:
    content: str
    doc_id: str = ""
    title: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: Optional[datetime] = field(default_factory=datetime.now)

@dataclass
class DocumentChunk:
    content: str
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict=field(default_factory=dict)
    embedding: Optional[list[float]] = None

def chunk_document(doc: Document, chunk_size: int = 900, overlap: int = 150):
    chunks = []
    text = doc.content
    start = 0

    while (start < len(text)):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunks.append(
            DocumentChunk(
                content=chunk_text,
                chunk_id=f"{doc.doc_id}_chunk_{len(chunks)}",
                doc_id=doc.doc_id,
                chunk_index=len(chunks),
                metadata={**doc.metadata, "chunk_size": chunk_size, "overlap": overlap}
            )
        )

        start += chunk_size - overlap
        if start >= len(text):
            break

    return chunks
