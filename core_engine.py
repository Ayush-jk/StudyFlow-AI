from groq import Groq
import re, json, numpy as np
from typing import List, Dict, Optional, Tuple

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SBERT_MODEL        = "all-MiniLM-L6-v2"
CROSS_ENC_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _extract_json(text: str):
    for pattern in [r"```(?:json)?\s*(\{.*?\})\s*```", r"\{.*\}"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            raw = m.group(1) if m.lastindex else m.group()
            for attempt in [raw, re.sub(r",\s*([}\]])", r"\1", raw)]:
                try: return json.loads(attempt)
                except: pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED RAG — Dense Retrieval · Hybrid Retrieval · Re-ranking · Query Expansion
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedRAGRetriever:
    """
    Full Advanced RAG pipeline:
      1. Chunking          — sliding-window with overlap
      2. Dense index       — SBERT (all-MiniLM-L6-v2) + numpy cosine
      3. Sparse index      — BM25Okapi (rank-bm25)
      4. Hybrid fusion     — Reciprocal Rank Fusion of dense + sparse
      5. Query Expansion   — LLM generates 3 alternative query phrasings
      6. Re-ranking        — CrossEncoder (ms-marco-MiniLM-L-6-v2)
    """

    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 60):
        self.chunk_size        = chunk_size
        self.chunk_overlap     = chunk_overlap
        self.chunks: List[Tuple[str, dict]] = []
        self.chunk_embeddings               = None   # np.ndarray (N, D)
        self.bm25                           = None
        self._bi_encoder                    = None   # SentenceTransformer
        self._cross_encoder                 = None   # CrossEncoder
        self.ready                          = False

    # ── lazy model loaders ────────────────────────────────────────────────────

    def _bi(self):
        if self._bi_encoder is None:
            from sentence_transformers import SentenceTransformer
            self._bi_encoder = SentenceTransformer(SBERT_MODEL)
        return self._bi_encoder

    def _ce(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(CROSS_ENC_MODEL)
        return self._cross_encoder

    # ── chunking ──────────────────────────────────────────────────────────────

    def _chunk(self, text: str, source: str) -> List[Tuple[str, dict]]:
        words  = text.split()
        step   = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        for i in range(0, len(words), step):
            w = words[i: i + self.chunk_size]
            if len(w) < 15:
                continue
            chunks.append((" ".join(w), {"source": source, "idx": len(chunks)}))
        return chunks

    # ── index building ────────────────────────────────────────────────────────

    def build(self, texts: Dict[str, str]):
        """Chunk all texts; build BM25 + SBERT dense indexes."""
        from rank_bm25 import BM25Okapi
        self.chunks = []
        for fname, text in texts.items():
            self.chunks.extend(self._chunk(text, fname))
        if not self.chunks:
            return
        tokenized             = [c[0].lower().split() for c in self.chunks]
        self.bm25             = BM25Okapi(tokenized)
        texts_list            = [c[0] for c in self.chunks]
        self.chunk_embeddings = self._bi().encode(
            texts_list, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self.ready = True

    # ── query expansion (LLM) ────────────────────────────────────────────────

    def expand_query(self, query: str, client) -> List[str]:
        """Generate 3 alternative phrasings via LLM to maximise recall."""
        prompt = (
            "Generate 3 alternative phrasings of this search query to maximise recall. "
            "Return ONLY a JSON array of 3 strings — no extra text.\n"
            f"Query: {query}"
        )
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=180, temperature=0.45,
                messages=[{"role": "user", "content": prompt}])
            raw = r.choices[0].message.content.strip()
            m   = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    return [query] + [str(x) for x in parsed[:3]]
        except Exception:
            pass
        return [query]

    # ── hybrid retrieval (BM25 + SBERT + RRF) ────────────────────────────────

    def _hybrid_one(self, query: str, top_k: int = 25) -> List[Tuple[str, dict, float]]:
        bm25_scores  = self.bm25.get_scores(query.lower().split())
        bm25_ranked  = np.argsort(bm25_scores)[::-1]
        q_emb        = self._bi().encode([query], normalize_embeddings=True).astype("float32")
        dense_scores = (self.chunk_embeddings @ q_emb.T).squeeze()
        dense_ranked = np.argsort(dense_scores)[::-1]
        k            = 60   # RRF constant
        rrf: dict    = {}
        for rank, idx in enumerate(bm25_ranked[:60]):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(dense_ranked[:60]):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [(self.chunks[i][0], self.chunks[i][1], s) for i, s in ranked[:top_k]]

    # ── cross-encoder re-ranking ──────────────────────────────────────────────

    def rerank(self, query: str, candidates: List[Tuple], top_k: int = 6) -> List[Tuple]:
        if not candidates:
            return []
        try:
            pairs  = [(query, c[0]) for c in candidates]
            scores = self._ce().predict(pairs)
            order  = np.argsort(scores)[::-1]
            return [candidates[int(i)] for i in order[:top_k]]
        except Exception:
            return candidates[:top_k]

    # ── full pipeline ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, client, top_k: int = 6) -> List[Tuple[str, dict, float]]:
        """Query expansion → multi-query hybrid retrieval → cross-encoder re-ranking."""
        if not self.ready:
            return []
        queries            = self.expand_query(query, client)
        seen, candidates   = set(), []
        for q in queries:
            for chunk_text, meta, score in self._hybrid_one(q, top_k=20):
                if chunk_text not in seen:
                    seen.add(chunk_text)
                    candidates.append((chunk_text, meta, score))
        return self.rerank(query, candidates, top_k=top_k)


# ══════════════════════════════════════════════════════════════════════════════
# LLM EVALUATION — DeepEval (faithfulness · answer relevancy · contextual relevancy)
# ══════════════════════════════════════════════════════════════════════════════

class StudyFlowEvaluator:
    """
    Evaluates every chat response using DeepEval metrics backed by the
    project's own Groq LLM.  Falls back to a direct Groq prompt-judge
    if DeepEval import fails.

    Metrics measured:
      • Faithfulness        — are all claims grounded in the retrieved context?
      • Answer Relevancy    — does the response actually answer the query?
      • Contextual Relevancy— are the retrieved chunks relevant to the query?
    """

    def __init__(self, client):
        self.client               = client
        self.eval_log: List[dict] = []
        self._deepeval_llm        = None
        self._use_deepeval        = False
        try:
            from deepeval.models.base_model import DeepEvalBaseLLM
            self._use_deepeval = True
            self._build_deepeval_llm(client, DeepEvalBaseLLM)
        except Exception:
            pass

    def _build_deepeval_llm(self, client, DeepEvalBaseLLM):
        _model = MODEL

        class _GroqDE(DeepEvalBaseLLM):
            def load_model(self_i):                  return client
            def generate(self_i, prompt, schema=None):
                r = client.chat.completions.create(
                    model=_model, max_tokens=600, temperature=0,
                    messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content
            async def a_generate(self_i, prompt, schema=None):
                return self_i.generate(prompt, schema)
            def get_model_name(self_i):              return f"groq/{_model}"

        self._deepeval_llm = _GroqDE()

    # ── main evaluate entry point ─────────────────────────────────────────────

    def evaluate(self, query: str, response: str, retrieved_chunks: List[str]) -> dict:
        if self._use_deepeval and self._deepeval_llm:
            result = self._evaluate_deepeval(query, response, retrieved_chunks)
        else:
            result = self._evaluate_fallback(query, response, retrieved_chunks)
        self.eval_log.append(result)
        return result

    def _evaluate_deepeval(self, query, response, retrieved_chunks) -> dict:
        try:
            from deepeval.metrics import (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualRelevancyMetric,
            )
            from deepeval.test_case import LLMTestCase

            context   = [c for c in retrieved_chunks[:6] if isinstance(c, str) and c.strip()]
            if not context:
                return self._evaluate_fallback(query, response, retrieved_chunks)

            test_case = LLMTestCase(input=query, actual_output=response,
                                    retrieval_context=context)
            scores, reasons = {}, {}
            for name, metric in [
                ("faithfulness",          FaithfulnessMetric(model=self._deepeval_llm, threshold=0.5, verbose_mode=False)),
                ("answer_relevancy",      AnswerRelevancyMetric(model=self._deepeval_llm, threshold=0.5, verbose_mode=False)),
                ("contextual_relevancy",  ContextualRelevancyMetric(model=self._deepeval_llm, threshold=0.5, verbose_mode=False)),
            ]:
                try:
                    metric.measure(test_case)
                    scores[name]  = round(float(metric.score), 3)
                    reasons[name] = getattr(metric, "reason", "")
                except Exception as e:
                    scores[name]  = None
                    reasons[name] = str(e)
            return {"query": query, "scores": scores, "reasons": reasons, "engine": "deepeval"}
        except Exception:
            return self._evaluate_fallback(query, response, retrieved_chunks)

    def _evaluate_fallback(self, query, response, retrieved_chunks) -> dict:
        """Direct Groq-judge evaluation (mirrors DeepEval scoring logic)."""
        context = "\n\n".join(c for c in retrieved_chunks[:4] if c.strip())[:2000]
        prompt  = f"""You are an LLM evaluation judge. Score this RAG response on 3 metrics (0.0–1.0).

Query: {query[:300]}
Retrieved Context: {context}
Response: {response[:600]}

Definitions:
- faithfulness: Every claim in the response is supported by the context. (1.0 = all claims grounded)
- answer_relevancy: The response directly answers the query. (1.0 = perfectly on-topic)
- contextual_relevancy: The retrieved context is relevant to the query. (1.0 = highly relevant)

Return ONLY this JSON (no markdown, no extra text):
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "contextual_relevancy": 0.0,
  "faithfulness_reason": "", "answer_relevancy_reason": "", "contextual_relevancy_reason": ""}}"""
        try:
            r    = self.client.chat.completions.create(
                model=MODEL, max_tokens=350, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            data = _extract_json(r.choices[0].message.content)
            if data:
                scores  = {k: data.get(k) for k in ["faithfulness", "answer_relevancy", "contextual_relevancy"]}
                reasons = {
                    "faithfulness":         data.get("faithfulness_reason", ""),
                    "answer_relevancy":     data.get("answer_relevancy_reason", ""),
                    "contextual_relevancy": data.get("contextual_relevancy_reason", ""),
                }
                return {"query": query, "scores": scores, "reasons": reasons, "engine": "groq-judge"}
        except Exception:
            pass
        return {"query": query, "scores": {"faithfulness": None, "answer_relevancy": None,
                "contextual_relevancy": None}, "reasons": {}, "engine": "failed"}

    # ── aggregation ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.eval_log:
            return {}
        keys = ["faithfulness", "answer_relevancy", "contextual_relevancy"]
        agg  = {k: [] for k in keys}
        for entry in self.eval_log:
            for k in keys:
                v = entry.get("scores", {}).get(k)
                if v is not None:
                    agg[k].append(float(v))
        return {k: round(sum(vs) / len(vs), 3) if vs else None for k, vs in agg.items()}

    def clear(self):
        self.eval_log.clear()


# ══════════════════════════════════════════════════════════════════════════════
# CORE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CodeTheoryWeaver:
    def __init__(self, api_key: str):
        self.client            = Groq(api_key=api_key)
        self.pdfs:             Dict[str, str] = {}
        self.codes:            Dict[str, str] = {}
        self.code_types:       Dict[str, str] = {}
        self.pdf_index:        Dict[str, str] = {}
        self.concept_clusters: List[dict]     = []
        self.links:            List[dict]     = []
        self.chat_history:     List[dict]     = []
        self.interaction_log:  List[str]      = []
        self.last_error:       str            = ""
        # FIX 3: Socratic state tracking
        self.socratic_resolved_topics: List[str] = []
        self.socratic_current_topic:   str        = ""
        # Advanced RAG
        self.retriever  = AdvancedRAGRetriever()
        self.rag_ready  = False
        # LLM Evaluation (DeepEval)
        self.evaluator  = StudyFlowEvaluator(self.client)

    # ── file ingestion ──────────────────────────────────────────────────────

    def add_pdf(self, filename: str, text: str):
        self.pdfs[filename] = text

    def add_code(self, filename: str, text: str):
        self.codes[filename]      = text
        self.code_types[filename] = self._detect_code_type(text)

    def inject_pseudocode(self, filename: str, text: str):
        self.codes[filename]      = text
        self.code_types[filename] = "implementation"

    def get_setup_scripts(self) -> List[str]:
        return [fn for fn, t in self.code_types.items() if t == "setup_script"]

    def _detect_code_type(self, code_text: str) -> str:
        setup_kw = [
            "pip install", "git clone", "wget ", "apt-get", "!pip", "!git",
            "subprocess.Popen", "localtunnel", "cloudflared", "npm install",
            "chmod +x", "tunnel", "ngrok", "trycloudflare", "streamlit run",
        ]
        setup_hits = sum(1 for kw in setup_kw if kw.lower() in code_text.lower())
        return "setup_script" if setup_hits >= 2 else "implementation"

    def clear(self):
        self.pdfs.clear(); self.codes.clear(); self.code_types.clear()
        self.pdf_index.clear()
        self.concept_clusters.clear(); self.links.clear()
        self.chat_history.clear(); self.interaction_log.clear()
        self.last_error = ""
        self.socratic_resolved_topics = []
        self.socratic_current_topic   = ""
        self.retriever  = AdvancedRAGRetriever()
        self.rag_ready  = False
        self.evaluator.clear()

    # ── Advanced RAG — index + context retrieval ────────────────────────────

    def build_rag_index(self):
        """Build dense (SBERT) + sparse (BM25) index over all uploaded content."""
        all_texts = {**self.pdfs, **self.codes}
        if all_texts:
            self.retriever.build(all_texts)
            self.rag_ready = self.retriever.ready

    def _rag_context(self, query: str, max_chars: int = 5000) -> Tuple[str, List[str]]:
        """
        Advanced RAG retrieval pipeline:
        Query Expansion → Hybrid (BM25 + SBERT) → Re-ranking → assembled context.
        Returns (context_string, list_of_raw_chunks).
        Falls back to full-context if index not ready.
        """
        if not self.rag_ready:
            return self._build_context(max_pdf=4500, max_code=1500), []
        results = self.retriever.retrieve(query, self.client, top_k=6)
        if not results:
            return self._build_context(max_pdf=4500, max_code=1500), []
        parts, chunks, total = [], [], 0
        for chunk_text, meta, _ in results:
            if total + len(chunk_text) > max_chars:
                break
            src = meta.get("source", "?")
            parts.append(f"[SOURCE: {src}]\n{chunk_text}")
            chunks.append(chunk_text)
            total += len(chunk_text)
        return "\n\n---\n\n".join(parts), chunks

    # ── FIX 2: Real line-number lookup ─────────────────────────────────────

    def get_lines(self, filename: str, start: int, end: int) -> str:
        """Return actual source lines from an uploaded code file."""
        if filename not in self.codes:
            return ""
        lines = self.codes[filename].split("\n")
        start = max(1, start)
        end   = min(len(lines), end)
        return "\n".join(lines[start - 1 : end])

    def _parse_line_range(self, code_lines_str: str):
        """Parse '45-52' or '45' into (45, 52)."""
        s = str(code_lines_str).strip()
        m = re.match(r"(\d+)\s*[-–]\s*(\d+)", s)
        if m:
            return int(m.group(1)), int(m.group(2))
        m2 = re.match(r"(\d+)", s)
        if m2:
            n = int(m2.group(1))
            return n, n + 10
        return None, None

    # ── PDF pre-indexing ────────────────────────────────────────────────────

    def _pre_index_pdf(self, filename: str, text: str) -> str:
        sample = text[:30000]
        prompt = f"""You are reading an academic or technical document. Extract structured facts.

Document text:
{sample}

Return ONLY valid JSON with these exact keys (no markdown, no preamble):
{{
  "title": "document title",
  "core_problem": "1-2 sentences: what problem or topic this document addresses",
  "central_insight": "1 sentence: the key idea or contribution",
  "key_quotes": ["up to 6 verbatim short quotes (under 15 words each) central to the argument"],
  "algorithm_steps": ["each step of the main algorithm or process described, in order"],
  "technical_components": ["every technical component, concept, model, or tool mentioned"],
  "key_metrics": ["each evaluation metric or measure mentioned with a one-phrase definition"],
  "main_results": ["key results or findings, with numbers where available"],
  "method_comparisons": ["any comparisons against other approaches with numbers"],
  "important_thresholds": ["any numerical thresholds, hyperparameters, or settings mentioned"]
}}"""
        try:
            r = self.client.chat.completions.create(
                model=MODEL, max_tokens=1500, temperature=0.05,
                messages=[{"role": "user", "content": prompt}])
            data = _extract_json(r.choices[0].message.content)
            if data:
                return json.dumps(data, indent=2)
        except Exception as e:
            self.last_error = f"Pre-index error: {e}"
        return ""

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE 1 — Semantic Concept Clustering
    # ══════════════════════════════════════════════════════════════════════════

    def create_bidirectional_links(self) -> list:
        self.last_error = ""

        # FIX 1: Require both PDF and code only for clustering, not for the whole app
        if not self.pdfs:
            self.last_error = "Need at least one PDF uploaded."
            return []
        if not self.codes:
            self.last_error = "Need at least one code file uploaded."
            return []

        for fname, text in self.pdfs.items():
            if fname not in self.pdf_index or not self.pdf_index[fname]:
                self.pdf_index[fname] = self._pre_index_pdf(fname, text)

        pdf_text  = self._extract_pdf_sections(max_chars=15000)
        code_text = self._extract_code_with_lines(max_chars_per_file=6000)

        index_block = "\n\n".join(
            f"[STRUCTURED INDEX: {fn}]\n{idx}"
            for fn, idx in self.pdf_index.items() if idx
        )
        setup_note = ""
        setup_scripts = self.get_setup_scripts()
        if setup_scripts:
            setup_note = f"""
NOTE: The following code files are SETUP/RUNNER scripts, not algorithm implementations:
{setup_scripts}
Map them to the CONCEPTUAL PURPOSE they serve (what algorithm they run, what task they perform),
NOT to low-level tasks like "dependency management" or "environment setup".
"""

        # FIX 2: Prompt now demands EXACT multi-line snippets with verified line ranges
        prompt = f"""You are an expert academic analyst performing DEEP SEMANTIC concept mapping.

━━━ PRE-EXTRACTED STRUCTURED FACTS (ground truth — trust these) ━━━
{index_block}

━━━ FULL THEORY TEXT (from PDF) ━━━
{pdf_text}

━━━ CODE with LINE NUMBERS (from uploaded files) ━━━
{code_text}
{setup_note}
━━━ TASK ━━━
Identify exactly 8 DEEP CONCEPTUAL CLUSTERS based ONLY on what is actually in the uploaded files.

STRICT RULES:
1. Every cluster must represent a DEEP algorithmic, mathematical, or methodological concept
   that genuinely appears in BOTH the theory and the code.
   FORBIDDEN cluster topics: "environment setup", "dependency management", "data download",
   "file I/O", "library installation", "notebook structure". These are NOT concepts.
2. Base clusters ONLY on what is actually present in the uploaded files.
3. Each cluster must exist in BOTH theory AND code.
4. Many-to-many: one passage or code section can appear in multiple clusters.
5. Draw code evidence from ALL uploaded code files — not just the first one.

CRITICAL FOR CODE EVIDENCE:
- code_lines MUST reference actual line numbers visible in the numbered code above (e.g. "12-18").
- snippet MUST be the EXACT verbatim code text from those line numbers — copy it character-for-character.
- If the concept spans multiple lines (a loop body, a function, a class), include ALL relevant lines
  in the range. Do NOT truncate to one line. Show the complete logical unit.
- The snippet must DIRECTLY implement or relate to the theory — not just be nearby code.

Each cluster:
- concept_name: specific, meaningful (3-7 words)
- bridge: 1 sentence on WHY theory and code share this concept
- theory_evidences: 1-3 items: {{pdf_file, page (int), quote (verbatim ≤12 words), explanation}}
- code_evidences: 1-3 items: {{code_file, start_line (int), end_line (int), snippet (exact verbatim multi-line code), explanation}}

Return ONLY valid JSON:
{{
  "clusters": [
    {{
      "concept_name": "Example Concept Name",
      "bridge": "One sentence explaining why theory and code both address this concept.",
      "theory_evidences": [
        {{"pdf_file": "example.pdf", "page": 3, "quote": "short verbatim quote here", "explanation": "What this quote means in context."}}
      ],
      "code_evidences": [
        {{"code_file": "example.py", "start_line": 12, "end_line": 18, "snippet": "exact verbatim code from lines 12-18", "explanation": "What this code does and how it relates to the concept."}}
      ]
    }}
  ]
}}"""

        try:
            r = self.client.chat.completions.create(
                model=MODEL, max_tokens=4500, temperature=0.1,
                messages=[{"role": "user", "content": prompt}])
            raw  = r.choices[0].message.content.strip()
            data = _extract_json(raw)
            if data and "clusters" in data:
                raw_clusters = [c for c in data["clusters"] if isinstance(c, dict)]
                # FIX 2: Overwrite each snippet with the REAL lines from the file
                for cluster in raw_clusters:
                    for ce in cluster.get("code_evidences", []):
                        fname  = ce.get("code_file", "")
                        s_line = ce.get("start_line")
                        e_line = ce.get("end_line")
                        if not (s_line and e_line):
                            s_line, e_line = self._parse_line_range(ce.get("code_lines", ""))
                        if fname and s_line and e_line:
                            real_code = self.get_lines(fname, s_line, e_line)
                            if real_code.strip():
                                ce["snippet"]    = real_code
                                ce["start_line"] = s_line
                                ce["end_line"]   = e_line
                                ce["code_lines"] = f"{s_line}-{e_line}"
                self.concept_clusters = raw_clusters
                self.links = self._clusters_to_flat_links()
                if not self.concept_clusters:
                    self.last_error = f"LLM returned empty clusters.\nRaw:\n{raw[:600]}"
            else:
                self.last_error = f"Could not parse JSON.\nRaw:\n{raw[:800]}"
        except Exception as e:
            self.last_error = f"API error: {e}"

        return self.concept_clusters

    def _clusters_to_flat_links(self) -> list:
        flat = []
        for c in self.concept_clusters:
            for te in c.get("theory_evidences", []):
                for ce in c.get("code_evidences", []):
                    flat.append({
                        "concept":        c.get("concept_name", ""),
                        "bridge":         c.get("bridge", ""),
                        "theory_file":    te.get("pdf_file", ""),
                        "page":           te.get("page", "?"),
                        "theory_quote":   te.get("quote", ""),
                        "theory_explain": te.get("explanation", ""),
                        "code_file":      ce.get("code_file", ""),
                        "code_lines":     ce.get("code_lines", ""),
                        "code_snippet":   ce.get("snippet", ""),
                        "code_explain":   ce.get("explanation", ""),
                    })
        return flat

    def explain_concept_cluster(self, concept_name: str, cluster: dict) -> str:
        ctx = self._build_context(max_pdf=4500, max_code=1500)
        index_ctx = "\n\n".join(
            f"[STRUCTURED INDEX: {fn}]\n{idx}"
            for fn, idx in self.pdf_index.items() if idx
        )
        te_text = "\n".join(
            f"  [{e.get('pdf_file')}, p.{e.get('page')}] \"{e.get('quote','')}\""
            for e in cluster.get("theory_evidences", [])
        )
        ce_text = "\n".join(
            f"  [{e.get('code_file')}, lines {e.get('code_lines','')}]\n{e.get('snippet','')}"
            for e in cluster.get("code_evidences", [])
        )
        prompt = f"""Using ONLY the uploaded files below, deep-explain the concept cluster "{concept_name}".

PRE-INDEXED FACTS (use as ground truth):
{index_ctx}

FULL UPLOADED CONTENT:
{ctx}

Theory evidence for this cluster:
{te_text}

Code evidence for this cluster (exact lines from file):
{ce_text}

Write a 6-8 sentence explanation covering ALL of:
1. What this concept IS in plain English — no jargon.
2. How the theory formalises it: cite the PDF with page number and a short verbatim quote.
3. How the code implements or approximates it: cite file and line numbers.
4. Any simplifications, trade-offs, or gaps between theory and code.
5. Key insight: what a student MUST understand to use this correctly.

Ground every sentence in the uploaded materials. Do not use external knowledge."""
        return self._call(prompt, max_tokens=1200)

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE 2 — Dual-Mode Chat
    # FIX 3: Socratic mode now detects correct answers and stops gracefully
    # ══════════════════════════════════════════════════════════════════════════

    def chat(self, question: str, mode: str = "normal") -> str:
        # ── Advanced RAG: retrieve relevant passages (replaces brute-force injection) ──
        rag_ctx, retrieved_chunks = self._rag_context(question)

        index_ctx = "\n\n".join(
            f"[PRE-INDEXED FACTS: {fn}]\n{idx}"
            for fn, idx in self.pdf_index.items() if idx
        )

        content_manifest = ""
        if self.pdf_index:
            for fn, idx in self.pdf_index.items():
                try:
                    facts      = json.loads(idx)
                    components = facts.get("technical_components", [])
                    quotes     = facts.get("key_quotes", [])
                    content_manifest += (
                        f"\nConfirmed content in {fn}: "
                        f"technical components = {components}; "
                        f"key quotes present = {quotes[:3]}"
                    )
                except Exception:
                    pass

        materials_block = f"""PRE-INDEXED KEY FACTS (use as primary ground truth for all citations):
{index_ctx}

FULL UPLOADED TEXT:
{rag_ctx}"""

        if mode == "socratic":
            # FIX 3: Detect if the student's latest message is a correct answer to the
            # previous Socratic question. If so, confirm and stop asking.
            last_ai_question = ""
            if self.chat_history:
                for msg in reversed(self.chat_history):
                    if msg["role"] == "assistant":
                        last_ai_question = msg["content"]
                        break

            # Add 12 spaces (3 tabs) right before "system" here:
            system = f"""You are a  deeply thoughtful Socratic mentor.
Your only goal is to help the student truly understand the uploaded study materials
through their own thinking — never by giving them the answer directly.

UPLOADED MATERIALS:
{materials_block}

CONFIRMED CONTENT IN UPLOADED MATERIALS:
{content_manifest}

Previous AI question (if any): "{last_ai_question[:300]}"

══════════════════════════════════════════════
STRICT SOCRATIC RULES — FOLLOW EVERY ONE
══════════════════════════════════════════════

TONE:
- Use natural, flowing conversational language.
- Be detailed and thoughtful in your responses, but never explain the concept yourself and never give any answer or explanation. You can only ask question if the users answer is wrong.

STRUCTURE OF EVERY RESPONSE:
1. Start by acknowledging what the student just said or asked (very small).
2. Gently guide them by reflecting on one specific part of their thinking(the most relavent aspect of the concept).
3. End with exactly well-chosen question(1or2 or at max 3) that pushes them deeper.
   - The question must be natural, not a quiz.
   - Your question is thierguide so keep that in mind while framing the question.

WHEN THIS IS THE FIRST QUESTION (student is asking something new):
   Then ask focused question/questions that invites them to explore a specific idea from the materials(the most relavent).
   You may include one gentle, incomplete hint if needed (e.g., "Think about what happens
   when the model is asked to 'think step by step'...").

WHEN THE STUDENT GIVES A PARTIAL OR INCOMPLETE ANSWER WHICH IS ALSO CORRECT:
   Praise what they got right.
   Then ask question/questions(at max 2 or 3) that gently points toward the missing piece without revealing it.

WHEN THE STUDENT GIVES A CLEAR, CORRECT, OR COMPLETE UNDERSTANDING:
   Respond with genuine warmth and affirmation in one natural sentence.
   Then write exactly this on a new line:
   ✓ You've understood this.
   Give a small 2 line of what the concpept was / what was cover inther interaction about this concept.
   After that — STOP completely. Do not ask anything else unless they start a new topic.

NEVER:
- Give any direct definition, explanation, or summary or hint.
- Use bullet points, numbered lists, or headers.
- Reveal the full answer or do the thinking for them.

You are here to help the student discover the ideas themselves using only the uploaded PDF
and code notebooks."""

        else:
            system = f"""You are a precise academic study assistant for the uploaded documents and code.

RULES:
1. Answer directly and concisely — no filler phrases.
2. EVERY factual claim MUST include an exact citation:
   - PDF: (filename, page X) AND quote ≤10 words verbatim when possible
   - Code: (filename, lines X-Y OR function name)
3. When a concept appears in BOTH the PDF and code, explicitly connect them.
4. NEVER say content is missing when you can see it in the materials.
   The materials explicitly contain:{content_manifest}
5. Use the PRE-INDEXED FACTS as your primary anchor for accuracy.
6. If uncertain about a specific detail, say so precisely — but still give the best answer from the materials.
7. Use ONLY the uploaded materials — no external knowledge.

UPLOADED MATERIALS:
{materials_block}"""

        messages = [
            {"role": "system", "content": system},
            *self.chat_history[-8:],
            {"role": "user", "content": question}
        ]
        answer = self._call_messages(messages, max_tokens=1200)
        self.chat_history += [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer}
        ]
        self.interaction_log.append(f"[{mode.upper()}] Q: {question}\nA: {answer}")

        # ── DeepEval evaluation (normal mode — Socratic answers are questions, not facts) ──
        if mode == "normal":
            try:
                self.evaluator.evaluate(question, answer, retrieved_chunks)
            except Exception:
                pass   # evaluation never breaks the main flow

        return answer

    def clear_chat(self):
        self.chat_history.clear()
        self.socratic_resolved_topics = []
        self.socratic_current_topic   = ""
        self.evaluator.clear()

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE 3 — Study Review
    # ══════════════════════════════════════════════════════════════════════════

    def generate_study_review(self) -> dict:
        if not self.interaction_log and not self.links:
            return {
                "summary":        "No session yet.",
                "progress":       "—",
                "depth_score":    0,
                "weak_areas":     [],
                "strong_areas":   [],
                "misconceptions": [],
                "next_steps":     "Upload files and interact to generate a review.",
            }

        log_text  = "\n\n".join(self.interaction_log[-20:])
        link_text = json.dumps(self.links[:8], indent=2)
        index_ctx = "\n\n".join(
            f"[{fn}]\n{idx}" for fn, idx in self.pdf_index.items() if idx
        )

        prompt = f"""You are an expert academic learning coach for a study agent session.

Student files — PDFs: {list(self.pdfs.keys())}, Code: {list(self.codes.keys())}

GROUND TRUTH (pre-indexed document facts):
{index_ctx}

Discovered concept clusters:
{link_text}

Full student interaction log:
{log_text}

Analyse BOTH the student's questions AND the AI's answers.
- If the AI incorrectly said content was missing from materials when it was present, flag this as a system limitation (not a student weakness).
- Focus student weaknesses on topics the student did NOT ask about or answered incorrectly.
- Note if the student showed correct Socratic reasoning (got confirmed answers ✓).
- Be honest and specific, not vague or generic.

Return ONLY this JSON (no markdown, no preamble):
{{
  "summary": "2-3 sentences: what specific topics were covered in the session",
  "progress": "2 sentences: honest depth assessment with specific evidence",
  "depth_score": 72,
  "strong_areas": ["each concept the student clearly demonstrated understanding of"],
  "weak_areas": [
    {{
      "topic": "specific concept not covered or misunderstood",
      "evidence": "concrete evidence from the log",
      "suggestion": "specific actionable study step grounded in the uploaded materials"
    }}
  ],
  "misconceptions": ["any incorrect assumption the student expressed"],
  "next_steps": "3 specific ordered recommendations grounded in the uploaded files"
}}
depth_score: integer 0-100. Be honest."""

        try:
            r = self.client.chat.completions.create(
                model=MODEL, max_tokens=1400, temperature=0.2,
                messages=[{"role": "user", "content": prompt}])
            data = _extract_json(r.choices[0].message.content)
            if data:
                return data
        except Exception as e:
            self.last_error = f"Study review error: {e}"

        return {
            "summary":        "Review failed.",
            "progress":       "—",
            "depth_score":    0,
            "weak_areas":     [],
            "strong_areas":   [],
            "misconceptions": [],
            "next_steps":     "Retry.",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_code_with_lines(self, max_chars_per_file: int = 6000) -> str:
        """
        FIX 2: Return code with EXPLICIT line numbers so the LLM can cite them accurately.
        """
        parts = []
        for fname, code in self.codes.items():
            ctype = self.code_types.get(fname, "implementation")
            lines = code.split("\n")
            numbered_lines = "\n".join(f"{i+1:5d} | {line}" for i, line in enumerate(lines))
            header = f"# ── File: {fname}  [type: {ctype}]  ({len(lines)} lines total) ──"
            parts.append(f"{header}\n{numbered_lines[:max_chars_per_file]}")
        return "\n\n".join(parts)

    def _extract_code_concepts(self, max_chars_per_file: int = 5000) -> str:
        """Legacy method kept for explain_concept_cluster context building."""
        parts = []
        for fname, code in self.codes.items():
            ctype = self.code_types.get(fname, "implementation")
            parts.append(f"# ── File: {fname}  [detected type: {ctype}] ──")
            lines = code.split("\n")
            kept  = []
            for i, line in enumerate(lines):
                s = line.strip()
                if (s.startswith("import ") or s.startswith("from ")
                        or s.startswith("class ") or s.startswith("def ")
                        or s.startswith("#")
                        or "return " in s
                        or ("self." in s and len(s) < 100)
                        or ("=" in s and len(s) < 80 and not s.startswith("="))):
                    kept.append(f"{i+1:4d}  {line}")
            parts.append("\n".join(kept[:300])[:max_chars_per_file])
        return "\n\n".join(parts)

    def _extract_pdf_sections(self, max_chars: int = 15000) -> str:
        parts = []
        for fname, text in self.pdfs.items():
            cleaned = re.sub(r"\n{3,}", "\n\n", text)
            cleaned = "\n".join(
                l for l in cleaned.split("\n") if len(l.strip()) > 8
            )
            parts.append(f"[{fname}]\n{cleaned[:max_chars]}")
        return "\n\n".join(parts)

    def _build_context(self, max_pdf: int = 4500, max_code: int = 1500) -> str:
        parts  = [f"[PDF: {n}]\n{t[:max_pdf]}"  for n, t in self.pdfs.items()]
        parts += [
            f"[CODE ({self.code_types.get(n, '?')}): {n}]\n{t[:max_code]}"
            for n, t in self.codes.items()
        ]
        if self.concept_clusters:
            summary = [
                {"concept": c.get("concept_name"), "bridge": c.get("bridge")}
                for c in self.concept_clusters[:6]
            ]
            parts.append(f"[CONCEPT CLUSTERS]\n{json.dumps(summary, indent=2)}")
        return "\n\n".join(parts)

    def _call(self, prompt: str, max_tokens: int = 800) -> str:
        try:
            r = self.client.chat.completions.create(
                model=MODEL, max_tokens=max_tokens, temperature=0.3,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def _call_messages(self, messages: list, max_tokens: int = 1000) -> str:
        try:
            r = self.client.chat.completions.create(
                model=MODEL, max_tokens=max_tokens, temperature=0.4,
                messages=messages)
            return r.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
