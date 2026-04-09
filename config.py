"""
StudyFlowAI – Configuration
============================
All external choices are justified below. No unjustified dependency exists.

WHY GROQ (free-tier LLaMA-3.1-8B-Instant)?
  - Groq offers a completely free API tier with generous rate limits.
  - Inference speed: ~750 tok/s (vs ~20 tok/s running the same model locally on T4).
  - Running an 8B+ model on T4 16 GB requires INT4 quantization, complex setup,
    and still delivers slow, lower-quality outputs compared to Groq's hardware.
  - Competitors: OpenAI/Anthropic are paid. Google AI Studio free tier has daily caps.
    Groq is uniquely both free AND fast — ideal for a student project.

WHY sentence-transformers (all-MiniLM-L6-v2) locally on T4?
  - Completely free — no API call, no token cost.
  - all-MiniLM-L6-v2 is 22 MB and runs on T4 in <5 ms per batch encode.
  - OpenAI text-embedding-ada-002 costs $0.0001/1K tokens — repeated PDF indexing
    adds up fast. Sentence-Transformers gives comparable retrieval quality for free.

WHY FAISS (faiss-cpu)?
  - Gold standard for vector similarity search; used by Meta at billion scale.
  - faiss-cpu is simpler to install in Colab than faiss-gpu and is still fast
    enough for any document a student would upload (< 1000 chunks).
  - Alternatives: ChromaDB, Pinecone, Weaviate are all heavier or cloud-based.
    FAISS keeps everything local and offline-safe.

WHY PyMuPDF (fitz)?
  - Best-in-class PDF text extraction; handles multi-column, tables, scanned pages.
  - PyPDF2 and pdfplumber miss text in complex layouts. PyMuPDF does not.

WHY Gradio?
  - Works in Google Colab with share=True out of the box.
  - No server setup needed. One line to expose a public URL.
  - Streamlit requires a separate ngrok tunnel in Colab — Gradio does not.

WHY SM-2 Algorithm (implemented from scratch, no library)?
  - SM-2 is a ~100-line algorithm — zero reason to add a library dependency.
  - Fully transparent, customizable, and understood end-to-end.
"""

import os

# ── LLM (Groq / LLaMA-3.1) ──────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # Set in Colab via: os.environ["GROQ_API_KEY"] = "gsk_..."
GROQ_MODEL   = "llama-3.1-8b-instant"               # Free tier; swap to llama-3.1-70b-versatile for better quality

# ── Embeddings (local, sentence-transformers) ────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"                   # 22 MB, fast on T4

# ── RAG Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE   = 400   # tokens (words approximation); keeps context tight
CHUNK_OVERLAP = 60   # prevents answer from falling at chunk boundary
TOP_K        = 5     # top-k chunks retrieved per query
