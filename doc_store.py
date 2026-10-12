from doc_model import Document, DocumentChunk, chunk_document
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DocumentStore:
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, DocumentChunk] = field(default_factory=dict)

    def add(self, document: Document) -> str:
        if not document.doc_id:
            document.doc_id = f"DOC_{len(self.documents) + 1:04d}"
        self.documents[document.doc_id] = document

        chunks = chunk_document(document, chunk_size=150, overlap=30)
        for c in chunks:
            self.add_chunk(c)

        return document.doc_id

    def get(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            chunk_ids = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
            for cid in chunk_ids:
                del self.chunks[cid]
            return True
        return False

    def list_all(self) -> list[Document]:
        return list(self.documents.values())

    def add_chunk(self, chunk: DocumentChunk) -> str:
        if not chunk.chunk_id:
            chunk.chunk_id = f"chunk_{len(self.chunks) + 1:04d}"
        self.chunks[chunk.chunk_id] = chunk
        return chunk.chunk_id

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self.chunks.get(chunk_id)

    def get_chunks_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        return [c for c in self.chunks.values() if c.doc_id == doc_id]

    def count(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}
