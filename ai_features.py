"""
ai_features.py — All Generative AI features
=============================================
GenAI Concepts demonstrated:
  1. Prompt Engineering     — role-specific system prompts shape LLM behaviour
  2. Socratic Dialogue      — LLM is instructed never to answer directly, only guide
  3. Structured Generation  — JSON-mode prompting extracts machine-readable output
  4. Knowledge Graph (LLM)  — entity/relation extraction via constrained prompting
  5. SM-2 Spaced Repetition — adaptive scheduling driven by AI-generated cards
  6. Adaptive Feedback      — LLM analyses performance data to personalise coaching
"""

import json
import io
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")                    # headless rendering (no display needed)
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL


# ── Groq client ──────────────────────────────────────────────────────────────
_client = Groq(api_key=GROQ_API_KEY)


def llm_call(
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Single-turn call to Groq LLaMA."""
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict | list:
    """Strip markdown fences then parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # take the block between first and second fence
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SOCRATIC TUTOR
# ─────────────────────────────────────────────────────────────────────────────
SOCRATIC_SYSTEM = """You are a Socratic tutor. Your ONLY job is to guide the student
to discover the answer themselves — you must NEVER state the answer directly.

Rules:
• Ask 1–3 probing questions that lead the student one logical step closer.
• Reference specific terms/concepts from the document context when possible.
• Acknowledge what the student already knows before asking your question.
• Keep your response under 120 words.
• If the student's reasoning is wrong, gently surface the contradiction with a question.
"""


def socratic_tutor(context: str, question: str, history: list[list[str]]) -> str:
    history_text = "\n".join(
        f"Student: {h[0]}\nTutor: {h[1]}"
        for h in (history[-4:] if history else [])
    )
    user = f"""## Document Context
{context}

## Conversation so far
{history_text if history_text else '(start of conversation)'}

## Student's latest message
{question}

Guide the student using Socratic questioning only."""
    return llm_call(SOCRATIC_SYSTEM, user, temperature=0.8, max_tokens=200)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  QUIZ GENERATION
# ─────────────────────────────────────────────────────────────────────────────
QUIZ_SYSTEM = """You are an expert educational assessment designer.
Generate multiple-choice questions STRICTLY from the provided document content.
Return ONLY valid JSON — no preamble, no markdown fences, no extra text.

Schema:
{
  "questions": [
    {
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "A",
      "explanation": "one sentence referencing the document"
    }
  ]
}"""


def generate_quiz(context: str, n: int = 5) -> dict:
    user = f"Create {n} MCQ questions from this content:\n\n{context[:4000]}"
    try:
        raw = llm_call(QUIZ_SYSTEM, user, temperature=0.4, max_tokens=2500)
        return _parse_json(raw)
    except Exception as e:
        print(f"[Quiz] Parse error: {e}")
        return {"questions": []}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FLASHCARD GENERATION
# ─────────────────────────────────────────────────────────────────────────────
FLASHCARD_SYSTEM = """You are a flashcard designer.
Create concise, high-signal flashcards from the document.
Fronts should be a specific question or term. Backs should be a crisp definition/answer.
Return ONLY valid JSON — no preamble, no markdown fences.

Schema:
{
  "flashcards": [
    {"front": "Question or term", "back": "Answer or definition"}
  ]
}"""


def generate_flashcards(context: str, n: int = 10) -> dict:
    user = f"Create {n} flashcards from:\n\n{context[:3500]}"
    try:
        raw = llm_call(FLASHCARD_SYSTEM, user, temperature=0.4, max_tokens=2000)
        return _parse_json(raw)
    except Exception as e:
        print(f"[Flashcards] Parse error: {e}")
        return {"flashcards": []}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  KNOWLEDGE GRAPH  (Study Map)
# ─────────────────────────────────────────────────────────────────────────────
GRAPH_SYSTEM = """You are a knowledge-graph extractor.
Extract the most important concepts and how they relate to each other from the text.
Return ONLY valid JSON — no preamble, no fences.

Schema:
{
  "nodes": ["ConceptA", "ConceptB", ...],
  "edges": [["ConceptA", "ConceptB", "relationship label"], ...]
}

Constraints: max 12 nodes, max 16 edges. Short node names (≤4 words).
Short relationship labels (≤3 words)."""


def _render_graph(data: dict) -> Image.Image | None:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        return None

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        if len(e) >= 2:
            G.add_edge(e[0], e[1], label=e[2] if len(e) > 2 else "")

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.2)

    # Edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#58a6ff", alpha=0.6,
        arrows=True, arrowsize=18,
        width=1.8, connectionstyle="arc3,rad=0.08",
    )
    # Edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, ax=ax,
        font_color="#8b949e", font_size=7.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#161b22", alpha=0.7),
    )
    # Nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color="#238636", node_size=2400, alpha=0.92,
    )
    # Labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_color="white", font_size=8.5, font_weight="bold",
    )

    ax.set_title("Knowledge Graph — Study Map", color="#e6edf3", fontsize=14, pad=16)
    ax.axis("off")
    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def generate_study_map(context: str) -> Image.Image | None:
    user = f"Extract concepts and relationships from:\n\n{context[:2500]}"
    try:
        raw  = llm_call(GRAPH_SYSTEM, user, temperature=0.3, max_tokens=900)
        data = _parse_json(raw)
        return _render_graph(data)
    except Exception as e:
        print(f"[StudyMap] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SPACED REPETITION  (SM-2 algorithm, implemented from scratch)
# ─────────────────────────────────────────────────────────────────────────────
class SpacedRepetition:
    """
    SM-2 algorithm — the same algorithm behind Anki.
    Quality scores: 0=blackout, 1=wrong, 2=wrong but familiar,
                    3=correct hard, 4=correct, 5=correct easy.
    """

    def __init__(self):
        self._cards: dict[str, dict] = {}

    def load_flashcards(self, flashcards: list[dict]) -> int:
        now = datetime.now()
        for i, card in enumerate(flashcards):
            cid = f"fc_{i}_{len(self._cards)}"
            self._cards[cid] = {
                "front":       card["front"],
                "back":        card["back"],
                "ease":        2.5,      # EF (ease factor)
                "interval":    0,        # days until next review
                "repetitions": 0,        # consecutive correct answers
                "due":         now,
            }
        return len(self._cards)

    def due_cards(self) -> list[tuple[str, dict]]:
        now = datetime.now()
        return [(k, v) for k, v in self._cards.items() if v["due"] <= now]

    def review(self, card_id: str, quality: int) -> dict:
        """Update SM-2 state after a review. Returns updated card dict."""
        c = self._cards[card_id]
        if quality >= 3:                          # correct response
            if c["repetitions"] == 0:
                c["interval"] = 1
            elif c["repetitions"] == 1:
                c["interval"] = 6
            else:
                c["interval"] = max(1, round(c["interval"] * c["ease"]))
            c["repetitions"] += 1
        else:                                     # incorrect — reset
            c["repetitions"] = 0
            c["interval"]    = 1

        # Update ease factor
        c["ease"] = max(
            1.3,
            c["ease"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02),
        )
        c["due"] = datetime.now() + timedelta(days=c["interval"])
        return c

    def stats(self) -> dict:
        total = len(self._cards)
        due   = len(self.due_cards())
        return {"total": total, "due_now": due, "scheduled": total - due}
