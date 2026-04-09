"""
app.py — StudyFlowAI Gradio Interface
======================================
Run in Colab:
    import os
    os.environ["GROQ_API_KEY"] = "gsk_YOUR_KEY_HERE"   # paste your Groq key
    !python app.py

Or with notebook cells — see README block at the bottom.
"""

import os
import gradio as gr

# ── Guard: key must be set before imports that initialise the Groq client ─────
if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError(
        "\n\n  GROQ_API_KEY is not set!\n"
        "  In Colab, run:  os.environ['GROQ_API_KEY'] = 'gsk_...'\n"
        "  Get a free key at: https://console.groq.com\n"
    )

from rag_engine   import RAGEngine
from ai_features  import (
    socratic_tutor, generate_quiz, generate_flashcards,
    generate_study_map, SpacedRepetition,
)
from analytics    import LearningAnalytics

# ── Singleton state (module-level, shared across callbacks) ───────────────────
rag       = RAGEngine()
sr        = SpacedRepetition()
analytics = LearningAnalytics()

# Mutable state containers (Gradio callbacks are stateless)
_quiz      = {"questions": [], "idx": 0, "score": 0}
_flashcard = {"cards": [], "idx": 0, "flipped": False}


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Tab 1 : Upload ────────────────────────────────────────────────────────────
def cb_upload(pdf_file):
    if pdf_file is None:
        return "⚠️ No file selected."
    try:
        n = rag.build_index(pdf_file.name)
        return (
            f"✅ **PDF indexed successfully!**\n\n"
            f"- Document: `{pdf_file.name.split('/')[-1]}`\n"
            f"- Chunks created: **{n}**\n"
            f"- Embedding model: `all-MiniLM-L6-v2` (local T4)\n"
            f"- Vector index: `FAISS IndexFlatIP`\n\n"
            f"You can now use all tabs below 🎓"
        )
    except Exception as e:
        return f"❌ Error: {e}"


# ── Tab 2 : Socratic Tutor ────────────────────────────────────────────────────
def cb_chat(message, history):
    if not message.strip():
        return history, ""
    if not rag.ready:
        return history + [[message, "⚠️ Please upload a PDF first."]], ""

    context  = rag.get_context(message)
    response = socratic_tutor(context, message, history)
    analytics.log_question_asked()
    return history + [[message, response]], ""


# ── Tab 3 : Quiz ─────────────────────────────────────────────────────────────
def cb_start_quiz(n_q):
    if not rag.ready:
        return "⚠️ Upload a PDF first.", gr.update(visible=False), gr.update(visible=False)

    context = rag.get_broad_context(30)
    result  = generate_quiz(context, int(n_q))
    qs      = result.get("questions", [])

    if not qs:
        return "❌ Quiz generation failed. Try again.", gr.update(visible=False), gr.update(visible=False)

    _quiz["questions"] = qs
    _quiz["idx"]       = 0
    _quiz["score"]     = 0

    q = qs[0]
    text = (
        f"**Question 1 / {len(qs)}**\n\n"
        f"{q['question']}\n\n" +
        "\n".join(q["options"])
    )
    return text, gr.update(visible=True), gr.update(visible=True)


def cb_answer_quiz(answer):
    qs = _quiz["questions"]
    i  = _quiz["idx"]

    if not qs or i >= len(qs):
        return "Quiz complete!", gr.update(visible=False), gr.update(visible=False)

    q       = qs[i]
    correct = answer.strip().upper().startswith(q["answer"].upper())

    analytics.log_quiz_answer(q["question"], correct)

    result_label = "✅ Correct!" if correct else f"❌ Wrong! Correct answer: **{q['answer']}**"
    feedback = (
        f"{result_label}\n\n"
        f"*{q['explanation']}*\n\n---\n\n"
    )
    if correct:
        _quiz["score"] += 1
    _quiz["idx"] += 1

    if _quiz["idx"] < len(qs):
        q_next = qs[_quiz["idx"]]
        feedback += (
            f"**Question {_quiz['idx']+1} / {len(qs)}**\n\n"
            f"{q_next['question']}\n\n" +
            "\n".join(q_next["options"])
        )
        return feedback, gr.update(visible=True), gr.update(visible=True)
    else:
        pct = round(_quiz["score"] / len(qs) * 100)
        feedback += (
            f"🎉 **Quiz complete!**\n\n"
            f"Score: **{_quiz['score']} / {len(qs)}** ({pct}%)"
        )
        return feedback, gr.update(visible=False), gr.update(visible=False)


# ── Tab 4 : Flashcards ────────────────────────────────────────────────────────
def cb_gen_flashcards():
    if not rag.ready:
        return "⚠️ Upload a PDF first.", "", gr.update(visible=False)

    context = rag.get_broad_context(20)
    result  = generate_flashcards(context, 10)
    cards   = result.get("flashcards", [])

    if not cards:
        return "❌ Could not generate flashcards. Try again.", "", gr.update(visible=False)

    sr.load_flashcards(cards)
    _flashcard["cards"]   = cards
    _flashcard["idx"]     = 0
    _flashcard["flipped"] = False

    return (
        f"### 🃏 Card 1 / {len(cards)}\n\n**{cards[0]['front']}**",
        "",
        gr.update(visible=True),
    )


def cb_flip():
    cards = _flashcard["cards"]
    if not cards:
        return "No cards loaded.", ""
    card = cards[_flashcard["idx"]]
    _flashcard["flipped"] = not _flashcard["flipped"]
    front = f"### 🃏 Card {_flashcard['idx']+1} / {len(cards)}\n\n**{card['front']}**"
    back  = f"**Answer:** {card['back']}" if _flashcard["flipped"] else ""
    return front, back


def cb_next_card():
    cards = _flashcard["cards"]
    if not cards:
        return "No cards loaded.", ""
    analytics.log_flashcard_viewed()
    _flashcard["idx"]     = (_flashcard["idx"] + 1) % len(cards)
    _flashcard["flipped"] = False
    card = cards[_flashcard["idx"]]
    return (
        f"### 🃏 Card {_flashcard['idx']+1} / {len(cards)}\n\n**{card['front']}**",
        "",
    )


def cb_prev_card():
    cards = _flashcard["cards"]
    if not cards:
        return "No cards loaded.", ""
    _flashcard["idx"]     = (_flashcard["idx"] - 1) % len(cards)
    _flashcard["flipped"] = False
    card = cards[_flashcard["idx"]]
    return (
        f"### 🃏 Card {_flashcard['idx']+1} / {len(cards)}\n\n**{card['front']}**",
        "",
    )


# ── Tab 5 : Study Map ─────────────────────────────────────────────────────────
def cb_study_map():
    if not rag.ready:
        return None, "⚠️ Upload a PDF first."
    context = rag.get_broad_context(12)
    img     = generate_study_map(context)
    analytics.log_map_generated()
    if img is None:
        return None, "❌ Graph generation failed. Try again."
    return img, "✅ Knowledge graph generated from your document."


# ── Tab 6 : Analytics ────────────────────────────────────────────────────────
def cb_analytics():
    stats_md, ai_feedback = analytics.render_report()
    return stats_md, ai_feedback


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
body { font-family: 'Segoe UI', sans-serif; }
.gradio-container { max-width: 900px !important; margin: auto; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), css=CSS, title="StudyFlowAI") as demo:

    gr.Markdown(
        """# 🎓 StudyFlowAI
    **RAG-powered interactive learning from your PDFs**
    *Upload a PDF → Use all 5 AI-powered learning tools below*"""
    )

    # ── Tab 1: Upload ─────────────────────────────────────────────────────────
    with gr.Tab("📄 Upload PDF"):
        gr.Markdown(
            "Upload any academic PDF. The system will chunk it, embed it with "
            "`all-MiniLM-L6-v2` on your T4 GPU, and build a FAISS vector index."
        )
        pdf_file    = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_btn  = gr.Button("⚡ Process PDF", variant="primary", size="lg")
        upload_out  = gr.Markdown()
        upload_btn.click(cb_upload, inputs=pdf_file, outputs=upload_out)

    # ── Tab 2: Socratic Tutor ─────────────────────────────────────────────────
    with gr.Tab("💬 Socratic Tutor"):
        gr.Markdown(
            "Ask about any concept in your PDF. The AI **never gives the answer directly** — "
            "it guides you using Socratic questioning to build deeper understanding."
        )
        chatbot  = gr.Chatbot(height=420, label="Tutor Dialogue", bubble_full_width=False)
        with gr.Row():
            msg_box  = gr.Textbox(
                placeholder="e.g. 'What is RAG and how does it reduce hallucinations?'",
                label="Your question", scale=5,
            )
            send_btn = gr.Button("Ask →", variant="primary", scale=1)

        send_btn.click(cb_chat, [msg_box, chatbot], [chatbot, msg_box])
        msg_box.submit(cb_chat,  [msg_box, chatbot], [chatbot, msg_box])

    # ── Tab 3: Quiz ───────────────────────────────────────────────────────────
    with gr.Tab("🧪 Quiz"):
        gr.Markdown(
            "AI generates multiple-choice questions **strictly from your document** "
            "using structured JSON prompting. Type **A**, **B**, **C**, or **D** to answer."
        )
        n_slider    = gr.Slider(3, 10, value=5, step=1, label="Number of Questions")
        quiz_btn    = gr.Button("Generate Quiz", variant="primary")
        quiz_out    = gr.Markdown()
        ans_box     = gr.Textbox(placeholder="A / B / C / D", label="Your Answer", visible=False)
        submit_btn  = gr.Button("Submit Answer", variant="secondary", visible=False)

        quiz_btn.click(cb_start_quiz, n_slider, [quiz_out, ans_box, submit_btn])
        submit_btn.click(cb_answer_quiz, ans_box, [quiz_out, ans_box, submit_btn])
        ans_box.submit(cb_answer_quiz,   ans_box, [quiz_out, ans_box, submit_btn])

    # ── Tab 4: Flashcards ─────────────────────────────────────────────────────
    with gr.Tab("🃏 Flashcards + Spaced Repetition"):
        gr.Markdown(
            "AI creates flashcards from your document. Cards are scheduled using the "
            "**SM-2 spaced-repetition algorithm** (same as Anki) — cards you struggle with "
            "appear more frequently."
        )
        gen_fc_btn  = gr.Button("Generate Flashcards", variant="primary")
        card_front  = gr.Markdown()
        card_back   = gr.Markdown()
        with gr.Row():
            prev_btn = gr.Button("← Prev")
            flip_btn = gr.Button("🔄 Flip", visible=False)
            next_btn = gr.Button("Next →")

        gen_fc_btn.click(cb_gen_flashcards, outputs=[card_front, card_back, flip_btn])
        flip_btn.click(cb_flip,      outputs=[card_front, card_back])
        next_btn.click(cb_next_card, outputs=[card_front, card_back])
        prev_btn.click(cb_prev_card, outputs=[card_front, card_back])

    # ── Tab 5: Study Map ──────────────────────────────────────────────────────
    with gr.Tab("🗺️ Knowledge Graph"):
        gr.Markdown(
            "The LLM extracts key concepts and their relationships from your document, "
            "then renders them as an interactive **directed knowledge graph**."
        )
        map_btn     = gr.Button("Generate Knowledge Graph", variant="primary")
        map_status  = gr.Markdown()
        map_img     = gr.Image(label="Knowledge Graph", type="pil")

        map_btn.click(cb_study_map, outputs=[map_img, map_status])

    # ── Tab 6: Analytics ──────────────────────────────────────────────────────
    with gr.Tab("📈 Analytics"):
        gr.Markdown(
            "Session performance is tracked across all tools. "
            "The **AI Coach** analyses your weak areas and gives personalised tips."
        )
        analytics_btn  = gr.Button("📊 Generate Report", variant="primary")
        stats_out      = gr.Markdown()
        gr.Markdown("### 🧠 AI Coach Feedback")
        feedback_out   = gr.Markdown()

        analytics_btn.click(cb_analytics, outputs=[stats_out, feedback_out])


# ══════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        share=True,       # generates a public gradio.live URL — perfect for Colab
        debug=True,
        show_error=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# COLAB NOTEBOOK QUICK-START (paste these cells in order)
# ══════════════════════════════════════════════════════════════════════════════
"""
# Cell 1 — Install dependencies
!pip install groq gradio sentence-transformers faiss-cpu PyMuPDF networkx matplotlib Pillow -q

# Cell 2 — Set your Groq API key  (free at https://console.groq.com)
import os
os.environ["GROQ_API_KEY"] = "gsk_YOUR_KEY_HERE"

# Cell 3 — Run the app
!python app.py
"""
