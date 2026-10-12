from doc_model import Document, DocumentChunk
from doc_store import DocumentStore
from embedding_similarity_helpers import embed_text, batch_embed_text, cosine_similarity
from dataclasses import dataclass, asdict
import json
from datetime import datetime

@dataclass
class VectorStore(DocumentStore):
    def add_chunk(self, chunk):
        if chunk.embedding is None:
            chunk.embedding = embed_text(chunk.content)
        return super().add_chunk(chunk)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[tuple[DocumentChunk, float]]:
        query_vec = embed_text(query)
        results = []

        for c in self.chunks.values():
            if c.embedding is None:
                continue
            score = cosine_similarity(query_vec, c.embedding)
            if score >= min_score:
                results.append((c, score))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def batch_search(self, query: list[str], top_k: int = 3, min_score: float = 0.0) -> list[list[tuple[DocumentChunk, float]]]:
        query_vecs = batch_embed_text(query)
        results_list = []

        for q_vec in query_vecs:
            results = []
            for c in self.chunks.values():
                if c.embedding is None:
                    continue
                score = cosine_similarity(q_vec, c.embedding)
                if score >= min_score:
                    results.append((c, score))
            results.sort(key=lambda x: x[1], reverse=True)
            results_list.append(results[:top_k])

        return results_list

    def save(self, path: str) -> None:
        documents = {}
        for doc_id, doc in self.documents.items():
            d = asdict(doc)
            d["created_at"] = doc.created_at.isoformat() if doc.created_at else None
            documents[doc_id] = d

        chunks = {}
        for chunk_id, chunk in self.chunks.items():
            c = asdict(chunk)
            # c.pop("embedding", None)
            chunks[chunk_id] = c

        store = {"documents": documents, "chunks": chunks}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)

        self.documents = {
            doc_id: Document(
                content=d["content"],
                doc_id=d["doc_id"],
                title=d["title"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=datetime.fromisoformat(d["created_at"]) if d["created_at"] else None
            ) for doc_id, d in store["documents"].items()
        }

        self.chunks = {
            chunk_id: DocumentChunk(
                content=c["content"],
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                chunk_index=c["chunk_index"],
                metadata=c["metadata"],
                embedding=c.get("embedding"),
            ) for chunk_id, c in store["chunks"].items()
        }