import streamlit as st
import PyPDF2
import json
import os
from core_engine import CodeTheoryWeaver

# ── FIX 5: Renamed to StudyFlow-AI everywhere ────────────────────────────────
st.set_page_config(
    page_title="StudyFlow-AI",
    page_icon="🎓",
    layout="wide"
)

ALLOWED_CODE_EXTS = {
    ".py", ".js", ".java", ".cpp", ".c", ".ts", ".go",
    ".rs", ".rb", ".cs", ".kt", ".ipynb"
}

API_KEY = os.environ.get("GROQ_API_KEY", "")


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_ipynb(file_bytes: bytes) -> str:
    try:
        nb    = json.loads(file_bytes.decode("utf-8", errors="replace"))
        cells = nb.get("cells", [])
        parts = []
        for i, cell in enumerate(cells):
            ctype  = cell.get("cell_type", "")
            source = cell.get("source", [])
            text   = "".join(source) if isinstance(source, list) else source
            if ctype == "code" and text.strip():
                parts.append(f"# ── Cell {i+1} ──\n{text}")
            elif ctype == "markdown" and text.strip():
                commented = "\n".join(f"# {l}" for l in text.split("\n"))
                parts.append(f"# ── Markdown {i+1} ──\n{commented}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"# Could not parse notebook: {e}"


def _init():
    defaults = {
        "weaver":              None,
        "chat_history":        [],
        "clusters":            [],
        "files_ready":         False,
        "chat_mode":           "normal",
        "last_review":         None,
        "link_error":          "",
        "pseudocode_injected": False,
        "pdf_count":           0,
        "code_count":          0,
        "eval_history":        [],   # per-turn DeepEval scores
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()

# ── CUSTOM CSS — StudyFlow-AI dark theme ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0e1220 !important;
    border-right: 1px solid #252d45;
}
section[data-testid="stSidebar"] * { color: #e2e8f8 !important; }

/* Main background */
.stApp { background: #080b12 !important; }

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
h1 { color: #00d4aa !important; letter-spacing: -0.5px; }
h3 { color: #9580ff !important; }

/* Cards / containers */
.element-container .stAlert { border-radius: 8px !important; }

/* Code blocks */
code, pre, [class*="stCode"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12.5px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #8892a4 !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom-color: #00d4aa !important;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #9580ff 0%, #6e4dff 100%) !important;
    border: none !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 20px rgba(149,128,255,.4) !important;
    transform: translateY(-1px) !important;
}

/* Secondary buttons */
.stButton > button[kind="secondary"] {
    background: #141828 !important;
    border: 1px solid #252d45 !important;
    color: #8892a4 !important;
    border-radius: 6px !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #141828 !important;
    border-radius: 8px !important;
    color: #e2e8f8 !important;
    font-weight: 600 !important;
}

/* Dividers */
hr { border-color: #252d45 !important; }

/* Metric */
[data-testid="metric-container"] {
    background: #141828 !important;
    border: 1px solid #252d45 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

/* Chat message styling */
.chat-user {
    background: linear-gradient(135deg, #9580ff, #6e4dff);
    color: white;
    padding: 12px 16px;
    border-radius: 10px 10px 3px 10px;
    margin: 8px 0 8px 40px;
    font-size: 14px;
    line-height: 1.6;
}
.chat-ai {
    background: #141828;
    border: 1px solid #252d45;
    color: #e2e8f8;
    padding: 12px 16px;
    border-radius: 10px 10px 10px 3px;
    margin: 8px 40px 8px 0;
    font-size: 14px;
    line-height: 1.6;
}
.chat-ai-socratic {
    background: rgba(245,166,35,0.07);
    border: 1px solid rgba(245,166,35,0.3);
    color: #e2e8f8;
    padding: 12px 16px;
    border-radius: 10px 10px 10px 3px;
    margin: 8px 40px 8px 0;
    font-size: 14px;
    line-height: 1.6;
}
.mode-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 99px;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-normal { background: rgba(0,212,170,0.15); color: #00d4aa; }
.badge-socratic { background: rgba(245,166,35,0.15); color: #f5a623; }

/* Score bar */
.score-bar-wrap {
    background: #1c2235;
    border-radius: 99px;
    height: 10px;
    width: 100%;
    margin: 6px 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 10px;
    border-radius: 99px;
    background: linear-gradient(90deg, #9580ff, #00d4aa);
    transition: width 0.5s ease;
}

/* Eval micro-badge */
.eval-chip {
    display: inline-block;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 7px;
    border-radius: 99px;
    margin: 2px 3px 2px 0;
    font-weight: 600;
}
.eval-good   { background: rgba(35,209,139,0.15); color: #23d18b; border: 1px solid rgba(35,209,139,0.3); }
.eval-mid    { background: rgba(245,166,35,0.15);  color: #f5a623; border: 1px solid rgba(245,166,35,0.3); }
.eval-bad    { background: rgba(241,76,76,0.15);   color: #f14c4c; border: 1px solid rgba(241,76,76,0.3); }
.eval-none   { background: rgba(136,146,164,0.1);  color: #8892a4; border: 1px solid rgba(136,146,164,0.2); }
</style>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 StudyFlow-AI")
    st.caption("Your AI-powered study agent")
    st.divider()

    if API_KEY:
        st.success("✅ Groq API key loaded")
    else:
        st.error("❌ No API key found — set GROQ_API_KEY in Colab and restart.")

    st.divider()

    if st.session_state.files_ready:
        w = st.session_state.weaver
        st.markdown("**📁 Loaded files**")
        for n in w.pdfs:
            st.markdown(f"- 📄 `{n}`")
        for n in w.codes:
            ctype = w.code_types.get(n, "?")
            icon  = "🔧" if ctype == "setup_script" else "💻"
            label = " *(setup)*" if ctype == "setup_script" else ""
            st.markdown(f"- {icon} `{n}`{label}")
        st.divider()

        col_a, col_b = st.columns(2)
        col_a.metric("📄 PDFs",  st.session_state.pdf_count)
        col_b.metric("💻 Files", st.session_state.code_count)
        if st.session_state.clusters:
            st.metric("🔗 Concept clusters", len(st.session_state.clusters))
        if w.pdf_index:
            st.success("✅ PDF pre-indexed")
        # Advanced RAG status
        if w.rag_ready:
            st.success("🔍 Advanced RAG active")
            st.caption(f"{len(w.retriever.chunks)} chunks indexed")
        # DeepEval status
        if st.session_state.eval_history:
            st.info(f"📊 {len(st.session_state.eval_history)} turns evaluated")
    else:
        st.info("Upload files in **Upload & Link** to begin.")

    st.divider()
    if st.button("🗑️ Reset session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── main area ─────────────────────────────────────────────────────────────────

st.markdown("# 🎓 StudyFlow-AI")
st.caption("Upload theory PDFs + code files → concept clusters · smart chat · study review")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📤 Upload & Link", "🔗 Concept Clusters", "💬 Chat", "📊 Study Review"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Link
# FIX 1: PDFs and code can be processed independently
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Upload your study materials")
    st.caption(
        "Upload **PDFs** for theory, **code files** for implementation — "
        "or both together for full concept-cluster mapping."
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📄 Theory PDFs")
        st.caption("Lecture notes, research papers, textbook chapters…")
        pdf_files = st.file_uploader(
            "Upload PDFs", type=["pdf"],
            accept_multiple_files=True, key="pdf_up",
            label_visibility="collapsed"
        )

    with col_r:
        st.subheader("💻 Code Files")
        st.caption("`.py` `.js` `.java` `.cpp` `.ts` `.go` `.ipynb` and more")
        code_files = st.file_uploader(
            "Upload Code Files",
            type=None,
            accept_multiple_files=True,
            key="code_up",
            label_visibility="collapsed",
            help="Accepts: " + ", ".join(sorted(ALLOWED_CODE_EXTS))
        )

    # Warn about unsupported code file types
    if code_files:
        bad = [f.name for f in code_files
               if not any(f.name.endswith(e) for e in ALLOWED_CODE_EXTS)]
        if bad:
            st.warning(
                f"⚠️ These files will be skipped (unsupported type): {', '.join(bad)}\n\n"
                f"Supported: {', '.join(sorted(ALLOWED_CODE_EXTS))}"
            )

    st.divider()

    # FIX 1: Allow processing with JUST PDFs or JUST code — not forced together
    has_pdfs  = bool(pdf_files)
    has_codes = bool(code_files and any(
        f.name.endswith(e) for f in code_files for e in ALLOWED_CODE_EXTS
    ))
    can_process = bool(API_KEY and (has_pdfs or has_codes))

    if has_pdfs and has_codes:
        st.info("✅ PDFs + code detected — full concept cluster mapping will be generated.")
    elif has_pdfs and not has_codes:
        st.info(
            "📄 PDF-only mode — Chat and Study Review will be fully available. "
            "Upload code files too to unlock **Concept Clusters**."
        )
    elif has_codes and not has_pdfs:
        st.info(
            "💻 Code-only mode — Chat and Study Review are available. "
            "Upload a PDF too to unlock **Concept Clusters**."
        )

    if not API_KEY:
        st.error("❌ API key not set. Run the API key cell in Colab first.")

    if st.button("🚀 Process Files", disabled=not can_process,
                 use_container_width=True, type="primary"):
        with st.spinner("Reading files and indexing…"):
            weaver = CodeTheoryWeaver(API_KEY)
            pdf_count  = 0
            code_count = 0

            for f in (pdf_files or []):
                try:
                    reader = PyPDF2.PdfReader(f)
                    text   = "\n".join(p.extract_text() or "" for p in reader.pages)
                    weaver.add_pdf(f.name, text)
                    st.caption(f"📄 Extracted {len(text):,} chars from {f.name}")
                    pdf_count += 1
                except Exception as e:
                    st.warning(f"Could not read {f.name}: {e}")

            for f in (code_files or []):
                if not any(f.name.endswith(e) for e in ALLOWED_CODE_EXTS):
                    continue
                try:
                    raw_bytes = f.read()
                    if f.name.endswith(".ipynb"):
                        code = extract_ipynb(raw_bytes)
                        st.caption(f"📓 Parsed notebook `{f.name}` — {len(code):,} chars")
                    else:
                        code = raw_bytes.decode("utf-8", errors="replace")
                        st.caption(f"💻 Loaded {len(code):,} chars from `{f.name}`")
                    weaver.add_code(f.name, code)
                    ctype = weaver.code_types.get(f.name, "?")
                    if ctype == "setup_script":
                        st.info(
                            f"🔧 `{f.name}` detected as a **setup/runner script** — "
                            "use the Inject Code panel below for deeper clusters."
                        )
                    code_count += 1
                except Exception as e:
                    st.warning(f"Could not read {f.name}: {e}")

            # FIX 1: Only build clusters if we have BOTH PDFs and code
            clusters   = []
            link_error = ""
            if weaver.pdfs and weaver.codes:
                with st.spinner("Building semantic concept clusters (this takes ~20s)…"):
                    clusters   = weaver.create_bidirectional_links()
                    link_error = weaver.last_error
            else:
                if weaver.pdfs and not weaver.codes:
                    # Still pre-index PDF so Chat works well
                    with st.spinner("Pre-indexing PDF for chat…"):
                        for fname, text in weaver.pdfs.items():
                            weaver.pdf_index[fname] = weaver._pre_index_pdf(fname, text)

            # ── Advanced RAG: build dense + BM25 index ────────────────────
            with st.spinner("🔍 Building Advanced RAG index (SBERT dense + BM25 sparse)…"):
                weaver.build_rag_index()
                if weaver.rag_ready:
                    st.caption(f"✅ RAG index ready — {len(weaver.retriever.chunks)} chunks")

            st.session_state.weaver               = weaver
            st.session_state.clusters             = clusters
            st.session_state.files_ready          = True
            st.session_state.chat_history         = []
            st.session_state.last_review          = None
            st.session_state.link_error           = link_error
            st.session_state.pseudocode_injected  = False
            st.session_state.pdf_count            = pdf_count
            st.session_state.code_count           = code_count
            st.session_state.eval_history         = []

        # Success messages
        if clusters:
            st.success(
                f"✅ Done! Found **{len(clusters)} concept clusters**. "
                "Head to the **Concept Clusters** tab →"
            )
        elif weaver.pdfs and weaver.codes:
            st.error("⚠️ Found 0 clusters. Check the Debug panel below.")
        elif weaver.pdfs:
            st.success(
                f"✅ {pdf_count} PDF(s) loaded and indexed. "
                "Head to **Chat** to start studying! Upload code files too to unlock Concept Clusters."
            )
        elif weaver.codes:
            st.success(
                f"✅ {code_count} code file(s) loaded. "
                "Head to **Chat** to start studying! Upload PDFs too to unlock Concept Clusters."
            )

    # ── debug panel ───────────────────────────────────────────────────────────
    if (st.session_state.files_ready
            and st.session_state.weaver
            and st.session_state.weaver.pdfs
            and st.session_state.weaver.codes
            and not st.session_state.clusters):
        with st.expander("🔍 Debug info", expanded=True):
            err = st.session_state.get("link_error", "")
            if err:
                st.error(f"**Engine error:**\n\n```\n{err}\n```")
            else:
                st.info("No error — LLM returned empty clusters array.")
            w = st.session_state.weaver
            if w:
                for n, t in w.pdfs.items():
                    st.markdown(f"**PDF preview — {n}:**"); st.code(t[:400])
                for n, t in w.codes.items():
                    st.markdown(f"**Code preview — {n}:**"); st.code(t[:400])
        if st.button("🔄 Retry Clustering"):
            with st.spinner("Retrying…"):
                clusters = st.session_state.weaver.create_bidirectional_links()
                st.session_state.clusters   = clusters
                st.session_state.link_error = st.session_state.weaver.last_error
            st.rerun()

    # ── code injection panel ──────────────────────────────────────────────────
    if st.session_state.files_ready:
        w = st.session_state.weaver
        setup_scripts = w.get_setup_scripts() if w else []

        st.divider()
        with st.expander(
            "💉 Inject Additional Code *(use when your code file is a setup script)*",
            expanded=bool(setup_scripts)
        ):
            if setup_scripts:
                st.warning(
                    f"⚠️ The following uploaded code files are **setup scripts**, not implementations:\n"
                    + "\n".join(f"- `{s}`" for s in setup_scripts)
                    + "\n\nPaste the core algorithm or implementation code below for richer clusters."
                )
            else:
                st.info("Optionally paste additional algorithm code to enrich concept clusters.")

            pseudo_name = st.text_input(
                "File name for injected code",
                value="injected_algorithm.py",
                key="pseudo_name"
            )
            pseudo_text = st.text_area(
                "Paste algorithm pseudocode or implementation code here",
                height=220,
                placeholder="def my_algorithm(inputs):\n    # core logic here\n    pass",
                key="pseudo_text"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💉 Inject & Re-link", type="primary",
                             use_container_width=True,
                             disabled=not pseudo_text.strip()):
                    w.inject_pseudocode(pseudo_name.strip(), pseudo_text.strip())
                    with st.spinner("Re-building concept clusters with injected code…"):
                        clusters = w.create_bidirectional_links()
                    # Rebuild RAG index to include injected code
                    with st.spinner("Updating RAG index…"):
                        w.build_rag_index()
                    st.session_state.clusters            = clusters
                    st.session_state.link_error          = w.last_error
                    st.session_state.pseudocode_injected = True
                    if clusters:
                        st.success(f"✅ Re-linked! {len(clusters)} clusters. Go to Concept Clusters tab →")
                    else:
                        st.error(f"Still 0 clusters. Error: {w.last_error}")
            with col_b:
                if st.session_state.pseudocode_injected:
                    st.success("✅ Code injected")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Concept Clusters
# FIX 2: Show REAL multi-line code from actual file, not LLM-hallucinated snippet
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.files_ready:
        st.info("👆 Upload and process files first.")
    elif not st.session_state.clusters:
        w = st.session_state.weaver
        if w and (not w.pdfs or not w.codes):
            st.warning(
                "🔗 Concept Clusters require **both** a PDF and a code file.\n\n"
                "You've only uploaded one type. Upload the other in the **Upload & Link** tab."
            )
        else:
            st.warning(
                "No clusters yet. Check the Debug panel in the Upload tab, "
                "or try injecting additional code."
            )
    else:
        clusters = st.session_state.clusters
        w        = st.session_state.weaver

        st.header(f"🔗 {len(clusters)} Semantic Concept Clusters")
        st.caption(
            "Each cluster = one deep concept that exists in BOTH theory and code. "
            "Code shown is extracted directly from your uploaded files."
        )

        if w and w.get_setup_scripts() and not st.session_state.pseudocode_injected:
            st.warning(
                "🔧 Some code files are setup scripts. Clusters may be shallow. "
                "Inject implementation code in the **Upload & Link** tab for deeper clusters."
            )

        st.divider()
        direction = st.radio("Explore direction:",
                             ["Theory → Code", "Code → Theory"], horizontal=True)
        st.divider()

        for idx, cluster in enumerate(clusters):
            concept = cluster.get("concept_name", f"Cluster {idx+1}")
            bridge  = cluster.get("bridge", "")
            t_evids = cluster.get("theory_evidences", [])
            c_evids = cluster.get("code_evidences", [])
            badge   = f"({len(t_evids)} theory · {len(c_evids)} code)"

            with st.expander(f"🧩 **{concept}** {badge}", expanded=False):
                st.info(f"🌉 **Why these connect:** {bridge}")
                st.divider()

                left_evids  = t_evids if direction == "Theory → Code" else c_evids
                right_evids = c_evids if direction == "Theory → Code" else t_evids
                left_label  = "📖 Theory Evidence" if direction == "Theory → Code" else "💻 Code Evidence"
                right_label = "💻 Code Evidence"   if direction == "Theory → Code" else "📖 Theory Evidence"

                col_l, col_r = st.columns(2)

                def render_evidence(evids, container, weaver_ref):
                    with container:
                        for e in evids:
                            if "pdf_file" in e:
                                st.markdown(f"📄 `{e.get('pdf_file')}` — page **{e.get('page','?')}**")
                                st.markdown(f"> *\"{e.get('quote','')}\"*")
                                st.caption(e.get("explanation", ""))
                            else:
                                fname      = e.get("code_file", "")
                                start_line = e.get("start_line")
                                end_line   = e.get("end_line")

                                # FIX 2: Parse start/end from string if ints not present
                                if not (start_line and end_line):
                                    cl = e.get("code_lines", "")
                                    m  = __import__("re").match(r"(\d+)\s*[-–]\s*(\d+)", str(cl))
                                    if m:
                                        start_line, end_line = int(m.group(1)), int(m.group(2))
                                    else:
                                        m2 = __import__("re").match(r"(\d+)", str(cl))
                                        if m2:
                                            n_ = int(m2.group(1))
                                            start_line, end_line = n_, n_ + 10

                                lines_label = (
                                    f"lines **{start_line}–{end_line}**"
                                    if start_line and end_line else "—"
                                )
                                st.markdown(f"💻 `{fname}` — {lines_label}")

                                # FIX 2: Pull real code from file, multi-line
                                if weaver_ref and fname and start_line and end_line:
                                    real_code = weaver_ref.get_lines(fname, start_line, end_line)
                                    if real_code.strip():
                                        ext  = fname.rsplit(".", 1)[-1].lower() if "." in fname else "python"
                                        lang = {
                                            "py": "python", "js": "javascript", "ts": "typescript",
                                            "java": "java", "cpp": "cpp", "c": "c", "go": "go",
                                            "rs": "rust", "rb": "ruby", "cs": "csharp",
                                        }.get(ext, "python")
                                        st.code(real_code, language=lang)
                                    else:
                                        snippet = e.get("snippet", "").strip()
                                        if snippet:
                                            st.code(snippet, language="python")
                                        else:
                                            st.caption("_(code snippet unavailable)_")
                                else:
                                    snippet = e.get("snippet", "").strip()
                                    if snippet:
                                        st.code(snippet, language="python")

                                st.caption(e.get("explanation", ""))
                            st.write("")

                with col_l: st.markdown(f"**{left_label}**")
                render_evidence(left_evids, col_l, w)
                with col_r: st.markdown(f"**{right_label}**")
                render_evidence(right_evids, col_r, w)

                if st.button(f"🔍 Deep-explain this concept", key=f"deep_{idx}"):
                    with st.spinner("Generating deep explanation…"):
                        deep = st.session_state.weaver.explain_concept_cluster(concept, cluster)
                    st.markdown("**📝 Deep Explanation:**")
                    st.markdown(deep)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Chat
# FIX 1: Available even without code files (PDF-only mode)
# FIX 3: Socratic mode detects correct answers and stops
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.files_ready:
        st.info("👆 Upload and process files first.")
    else:
        st.header("💬 Chat with your study materials")

        col_mode, col_clear = st.columns([4, 1])
        with col_mode:
            mode = st.radio(
                "Mode:", ["normal", "socratic"], horizontal=True,
                format_func=lambda m: (
                    "🎯 Normal — direct answers with citations" if m == "normal"
                    else "🤔 Socratic — guided questions only"
                )
            )
            st.session_state.chat_mode = mode
        with col_clear:
            st.write(""); st.write("")
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.eval_history = []
                st.session_state.weaver.clear_chat()
                st.rerun()

        # Mode banner
        if mode == "socratic":
            st.warning(
                "🤔 **Socratic mode** — I will only ask guiding questions. "
                "Once you arrive at the correct answer, I will confirm it and stop. "
                "No direct explanations."
            )
        else:
            st.success("🎯 **Normal mode** — direct, cited answers from your uploaded materials.")

        w = st.session_state.weaver
        loaded_info = []
        if w.pdfs:  loaded_info.append(f"{len(w.pdfs)} PDF(s)")
        if w.codes: loaded_info.append(f"{len(w.codes)} code file(s)")
        rag_label = " · 🔍 Advanced RAG active" if w.rag_ready else ""
        if loaded_info:
            st.caption(f"📚 Grounded in: {', '.join(loaded_info)}" +
                       ((" · ✅ PDF pre-indexed") if w.pdf_index else "") + rag_label)

        st.divider()

        # Input
        user_input = st.text_input(
            "Ask anything about your uploaded files:",
            placeholder=(
                "e.g. What is the main algorithm described in the PDF?"
                if mode == "normal"
                else "e.g. How does the loss function work?"
            ),
            key="chat_input"
        )

        if st.button("➤ Send", type="primary") and user_input.strip():
            with st.spinner("Thinking…"):
                answer = st.session_state.weaver.chat(user_input.strip(), mode)
            st.session_state.chat_history.append((user_input.strip(), answer, mode))
            # Capture the latest DeepEval result (if any) from the evaluator log
            eval_log = st.session_state.weaver.evaluator.eval_log
            if mode == "normal" and eval_log:
                st.session_state.eval_history.append(eval_log[-1])
            else:
                st.session_state.eval_history.append(None)
            st.rerun()

        st.divider()

        # ── helper: render DeepEval score chips ──────────────────────────────
        def _eval_chip(label, score):
            if score is None:
                return f'<span class="eval-chip eval-none">{label} —</span>'
            pct = int(score * 100)
            cls = "eval-good" if score >= 0.7 else ("eval-mid" if score >= 0.5 else "eval-bad")
            return f'<span class="eval-chip {cls}">{label} {pct}%</span>'

        # Render chat history — newest first
        history     = st.session_state.chat_history
        eval_hist   = st.session_state.eval_history
        if not history:
            st.caption("No messages yet. Ask a question above.")
        else:
            for i, (q, a, m) in enumerate(reversed(history)):
                rev_i      = len(history) - 1 - i
                eval_entry = eval_hist[rev_i] if rev_i < len(eval_hist) else None
                badge_cls  = "badge-normal" if m == "normal" else "badge-socratic"
                badge_text = "🎯 Normal" if m == "normal" else "🤔 Socratic"
                ai_cls     = "chat-ai" if m == "normal" else "chat-ai-socratic"

                # User message
                st.markdown(
                    f'<div class="chat-user">'
                    f'<span class="mode-badge {badge_cls}">{badge_text}</span><br>'
                    f'{q}</div>',
                    unsafe_allow_html=True
                )
                # AI message
                st.markdown(
                    f'<div class="{ai_cls}">'
                    f'<span class="mode-badge {badge_cls}">StudyFlow-AI</span><br>'
                    f'{a.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )
                # DeepEval micro-score chips (normal mode only)
                if m == "normal" and eval_entry and eval_entry.get("scores"):
                    sc   = eval_entry["scores"]
                    eng  = eval_entry.get("engine", "")
                    chips = (
                        _eval_chip("Faith", sc.get("faithfulness"))
                        + _eval_chip("Rel", sc.get("answer_relevancy"))
                        + _eval_chip("Ctx", sc.get("contextual_relevancy"))
                        + f'<span class="eval-chip eval-none" style="font-size:9px">{eng}</span>'
                    )
                    st.markdown(
                        f'<div style="margin: 2px 40px 10px 0; text-align:right">{chips}</div>',
                        unsafe_allow_html=True
                    )
                st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Study Review
# FIX 4: Study-agent framing throughout
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not st.session_state.files_ready:
        st.info("👆 Upload and process files first.")
    else:
        st.header("📊 Study Review")
        st.caption(
            "A personalised assessment of your session — what you studied, "
            "how well you engaged, where to focus next."
        )

        if st.button("📊 Generate My Study Review",
                     use_container_width=True, type="primary"):
            with st.spinner("Analysing your full session…"):
                rv = st.session_state.weaver.generate_study_review()
            st.session_state.last_review = rv

        if st.session_state.last_review:
            rv = st.session_state.last_review
            st.divider()

            # Summary + Progress
            col_sum, col_prog = st.columns(2)
            with col_sum:
                st.subheader("📚 What You Studied")
                st.info(rv.get("summary", "—"))
            with col_prog:
                st.subheader("📈 Learning Progress")
                st.info(rv.get("progress", "—"))

            # Depth score bar
            score = rv.get("depth_score", 0)
            score_color = (
                "#f14c4c" if score < 40
                else "#f5a623" if score < 70
                else "#23d18b"
            )
            st.subheader(f"🎯 Depth Score: {score}/100")
            st.markdown(
                f'<div class="score-bar-wrap">'
                f'<div class="score-bar-fill" style="width:{score}%;background:{score_color}"></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "Low (<40): surface level · Medium (40-70): solid foundation · High (70+): deep understanding"
            )

            st.divider()

            # ── DeepEval RAG Evaluation Summary ──────────────────────────────
            w = st.session_state.weaver
            eval_summary = w.evaluator.summary() if w else {}
            if eval_summary and any(v is not None for v in eval_summary.values()):
                st.subheader("🔬 RAG Quality Evaluation (DeepEval)")
                st.caption(
                    "Average scores across all Normal-mode chat turns — "
                    "measuring how well the retrieval system served your session."
                )
                ec1, ec2, ec3 = st.columns(3)
                def _score_display(container, label, icon, val):
                    if val is not None:
                        color = "#23d18b" if val >= 0.7 else ("#f5a623" if val >= 0.5 else "#f14c4c")
                        container.markdown(
                            f'<div style="background:#141828;border:1px solid #252d45;border-radius:8px;'
                            f'padding:14px;text-align:center">'
                            f'<div style="font-size:22px">{icon}</div>'
                            f'<div style="font-size:26px;font-weight:700;color:{color}">{int(val*100)}%</div>'
                            f'<div style="font-size:12px;color:#8892a4;margin-top:4px">{label}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        container.caption(f"{label}: no data")
                _score_display(ec1, "Faithfulness",        "🛡️", eval_summary.get("faithfulness"))
                _score_display(ec2, "Answer Relevancy",    "🎯", eval_summary.get("answer_relevancy"))
                _score_display(ec3, "Contextual Relevancy","🔍", eval_summary.get("contextual_relevancy"))
                st.caption(
                    "**Faithfulness** — claims grounded in retrieved context · "
                    "**Answer Relevancy** — response answers the query · "
                    "**Contextual Relevancy** — retrieved chunks matched the query"
                )

                # Per-turn detail (expandable)
                n_turns = len(w.evaluator.eval_log)
                if n_turns > 0:
                    with st.expander(f"📋 Per-turn evaluation log ({n_turns} turns)", expanded=False):
                        for j, entry in enumerate(w.evaluator.eval_log):
                            sc = entry.get("scores", {})
                            q  = entry.get("query", "")[:80]
                            faith = sc.get("faithfulness")
                            rel   = sc.get("answer_relevancy")
                            ctx   = sc.get("contextual_relevancy")
                            f_str = f"{int(faith*100)}%" if faith is not None else "—"
                            r_str = f"{int(rel*100)}%"   if rel   is not None else "—"
                            c_str = f"{int(ctx*100)}%"   if ctx   is not None else "—"
                            st.markdown(
                                f"**Turn {j+1}** `{q}…` — "
                                f"Faith **{f_str}** · Rel **{r_str}** · Ctx **{c_str}**"
                            )
                st.divider()

            # Strong vs Weak areas
            col_s, col_w = st.columns(2)
            with col_s:
                st.subheader("✅ Strong Areas")
                strong = rv.get("strong_areas", [])
                if strong:
                    for area in strong:
                        st.success(f"✅ {area}")
                else:
                    st.caption("Keep studying to build strong areas!")
            with col_w:
                st.subheader("⚠️ Needs More Work")
                weak = rv.get("weak_areas", [])
                if weak:
                    for item in weak:
                        st.error(f"**{item.get('topic', '')}**")
                        st.caption(f"📌 Evidence: {item.get('evidence', '')}")
                        st.warning(f"💡 Suggestion: {item.get('suggestion', '')}")
                else:
                    st.caption("No weak areas identified yet.")

            # Misconceptions
            miscon = rv.get("misconceptions", [])
            if miscon:
                st.divider()
                st.subheader("🚨 Misconceptions Detected")
                for m_item in miscon:
                    st.error(f"⚠️ {m_item}")

            # Next Steps
            st.divider()
            st.subheader("🗺️ Recommended Next Steps")
            st.warning(rv.get("next_steps", "—"))

        else:
            st.caption(
                "Hit **Generate My Study Review** above after studying with "
                "Concept Clusters and Chat to get your personalised assessment."
            )
