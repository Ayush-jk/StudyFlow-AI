"""
analytics.py — Session tracking + AI-generated personalised feedback
=====================================================================
GenAI Concept:
  • Adaptive Coaching — LLM analyses structured performance data and generates
    personalised, actionable study recommendations.
  • Weak-area detection — per-topic accuracy tracking flags knowledge gaps.
"""

import json
from collections import defaultdict
from ai_features import llm_call


class LearningAnalytics:
    def __init__(self):
        self._quiz_log:    list[dict]            = []   # {question, correct, topic}
        self._topic_hits:  dict[str, list[int]]  = defaultdict(list)
        self.counters = {
            "questions_asked":    0,
            "quiz_answers":       0,
            "correct_answers":    0,
            "flashcards_viewed":  0,
            "maps_generated":     0,
        }

    # ── Logging helpers ───────────────────────────────────────────────────────
    def log_quiz_answer(self, question: str, correct: bool, topic: str = "General") -> None:
        self._quiz_log.append({"q": question, "correct": correct, "topic": topic})
        self._topic_hits[topic].append(1 if correct else 0)
        self.counters["quiz_answers"]   += 1
        self.counters["correct_answers"] += int(correct)

    def log_question_asked(self)     -> None: self.counters["questions_asked"]   += 1
    def log_flashcard_viewed(self)   -> None: self.counters["flashcards_viewed"] += 1
    def log_map_generated(self)      -> None: self.counters["maps_generated"]    += 1

    # ── Derived metrics ───────────────────────────────────────────────────────
    def accuracy(self) -> float:
        total = self.counters["quiz_answers"]
        return round(self.counters["correct_answers"] / total * 100, 1) if total else 0.0

    def weak_areas(self, threshold: float = 0.60) -> list[str]:
        """Topics where accuracy < threshold."""
        weak = []
        for topic, scores in self._topic_hits.items():
            if scores and (sum(scores) / len(scores)) < threshold:
                weak.append(topic)
        return weak

    def summary(self) -> dict:
        return {
            "accuracy_pct":       self.accuracy(),
            "quiz_answers":       self.counters["quiz_answers"],
            "correct":            self.counters["correct_answers"],
            "questions_to_tutor": self.counters["questions_asked"],
            "flashcards_viewed":  self.counters["flashcards_viewed"],
            "maps_generated":     self.counters["maps_generated"],
            "weak_areas":         self.weak_areas(),
            "topic_breakdown":    {
                t: {
                    "attempts":  len(s),
                    "accuracy":  round(sum(s) / len(s) * 100, 1) if s else 0,
                }
                for t, s in self._topic_hits.items()
            },
        }

    # ── AI Coach Feedback ─────────────────────────────────────────────────────
    def ai_coaching_feedback(self) -> str:
        """Ask the LLM to generate personalised coaching tips from performance data."""
        data = self.summary()
        system = (
            "You are an expert learning coach specialising in evidence-based study techniques. "
            "Analyse the student's performance data and give exactly 3 concise, actionable tips. "
            "Each tip should be on its own line starting with a relevant emoji. "
            "Tone: encouraging but specific. Max 180 words total."
        )
        user = (
            f"Student performance summary:\n{json.dumps(data, indent=2)}\n\n"
            "Give 3 personalised improvement tips based on this data."
        )
        try:
            return llm_call(system, user, temperature=0.7, max_tokens=300)
        except Exception as e:
            return f"Could not generate feedback: {e}"

    # ── Markdown report ───────────────────────────────────────────────────────
    def render_report(self) -> tuple[str, str]:
        """Returns (stats_markdown, ai_feedback_text)."""
        s = self.summary()
        weak_str = ", ".join(s["weak_areas"]) if s["weak_areas"] else "None detected yet 🎉"

        topic_rows = "\n".join(
            f"| {t} | {v['attempts']} | {v['accuracy']}% |"
            for t, v in s["topic_breakdown"].items()
        ) or "| — | — | — |"

        stats_md = f"""## 📊 Your Learning Dashboard

| Metric | Value |
|--------|-------|
| Quiz Accuracy | **{s['accuracy_pct']}%** |
| Questions Answered | {s['quiz_answers']} ({s['correct']} correct) |
| Tutor Questions Asked | {s['questions_to_tutor']} |
| Flashcards Reviewed | {s['flashcards_viewed']} |
| Study Maps Generated | {s['maps_generated']} |
| Weak Areas | {weak_str} |

### Topic Breakdown
| Topic | Attempts | Accuracy |
|-------|----------|----------|
{topic_rows}
"""
        feedback = self.ai_coaching_feedback()
        return stats_md, feedback
