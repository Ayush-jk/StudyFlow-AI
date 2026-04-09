"""
rag_engine.py — Retrieval-Augmented Generation core
=====================================================
GenAI Concepts:
  1. Text Chunking     — splits document into semantically manageable pieces
  2. Dense Embeddings  — sentence-transformers encodes chunks into 384-dim vectors
  3. Vector Index      — FAISS IndexFlatIP (cosine via L2-normalised inner product)
  4. Semantic Retrieval— top-k nearest-neighbour search at query time
"""

import fitz                              # PyMuPDF
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K


class RAGEngine:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RAG] Loading embedder on {device} …")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        self.index    = None
        self.chunks   = []          # raw text chunks
        self.metadata = []          # (page_num, char_start) per chunk
        self.doc_name = ""

    # ── PDF → text ───────────────────────────────────────────────────────────
    def _load_pdf(self, path: str) -> tuple[str, list[dict]]:
        """Return full text and per-page info."""
        doc        = fitz.open(path)
        pages_data = []
        full_text  = ""
        for pno, page in enumerate(doc):
            text       = page.get_text("text")
            pages_data.append({"page": pno + 1, "text": text})
            full_text += text + "\n"
        doc.close()
        return full_text, pages_data

    # ── Text → overlapping word-level chunks ─────────────────────────────────
    def _chunk(self, text: str) -> list[str]:
        words  = text.split()
        step   = CHUNK_SIZE - CHUNK_OVERLAP
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if len(chunk.strip()) > 30:          # skip near-empty chunks
                chunks.append(chunk.strip())
        return chunks

    # ── Build FAISS index from PDF ────────────────────────────────────────────
    def build_index(self, pdf_path: str) -> int:
        """Process a PDF and build the vector store. Returns chunk count."""
        self.doc_name = pdf_path.split("/")[-1]
        full_text, _ = self._load_pdf(pdf_path)

        self.chunks = self._chunk(full_text)
        if not self.chunks:
            raise ValueError("No text could be extracted from this PDF.")

        print(f"[RAG] Encoding {len(self.chunks)} chunks …")
        embeddings = self.embedder.encode(
            self.chunks,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 norm → inner product == cosine sim
        )

        dim         = embeddings.shape[1]
        self.index  = faiss.IndexFlatIP(dim)   # Inner-Product index
        self.index.add(embeddings)
        print(f"[RAG] Index built: {self.index.ntotal} vectors, dim={dim}")
        return len(self.chunks)

    # ── Query → top-k chunks ─────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        if self.index is None:
            return []
        q_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, idxs = self.index.search(q_emb, top_k)
        return [
            (self.chunks[i], float(scores[0][j]))
            for j, i in enumerate(idxs[0])
            if 0 <= i < len(self.chunks)
        ]

    def get_context(self, query: str, top_k: int = TOP_K) -> str:
        """Concatenate retrieved chunks into a single context string."""
        results = self.retrieve(query, top_k)
        return "\n\n---\n\n".join(chunk for chunk, _ in results)

    def get_broad_context(self, n_chunks: int = 25) -> str:
        """Return the first n chunks — useful for quiz/flashcard generation."""
        return " ".join(self.chunks[:n_chunks])

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.chunks)
