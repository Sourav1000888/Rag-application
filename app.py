import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.readers.file import PyMuPDFReader

load_dotenv()

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="DocuMind · Chat with your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide default chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2rem; max-width: 980px; }

/* hero */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 55%, #7c3aed 100%);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    color: #fff;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 32px rgba(67, 56, 202, 0.25);
}
.hero h1 {
    font-family: 'Fraunces', serif;
    font-size: 2.3rem;
    margin: 0 0 .3rem 0;
    color: #fff;
}
.hero p { margin: 0; opacity: .85; font-size: 1.02rem; }

/* stat cards */
.stat {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 14px;
    padding: .9rem 1.1rem;
}
.stat .v { font-size: 1.5rem; font-weight: 600; color: #6366f1; }
.stat .l { font-size: .8rem; opacity: .7; }

/* source cards */
.source {
    border-left: 4px solid #6366f1;
    background: var(--secondary-background-color);
    border-radius: 10px;
    padding: .8rem 1rem;
    margin-bottom: .6rem;
    font-size: .88rem;
}
.source .meta { font-weight: 600; margin-bottom: .3rem; color: #6366f1; }
.badge {
    display: inline-block;
    background: rgba(99,102,241,.15);
    color: #6366f1;
    border-radius: 999px;
    padding: 1px 10px;
    font-size: .75rem;
    margin-left: .5rem;
}

/* empty state */
.empty {
    text-align: center;
    padding: 3rem 1rem;
    border: 2px dashed rgba(128,128,128,.3);
    border-radius: 18px;
    opacity: .85;
}
.empty .icon { font-size: 3rem; }

/* buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(99,102,241,.4);
    transition: all .15s ease;
}
.stButton > button:hover {
    border-color: #6366f1;
    color: #6366f1;
    transform: translateY(-1px);
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def configure_models(api_key: str, llm_model: str):
    Settings.embed_model = load_embedding_model()
    Settings.llm = OpenRouter(api_key=api_key, model=llm_model)


def build_index(uploaded_file):
    """Read the PDF, then chunk + embed + index it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        docs = PyMuPDFReader().load_data(file_path=path)
    finally:
        os.remove(path)
    index = VectorStoreIndex.from_documents(documents=docs)
    return index, len(docs)


def build_query_engine(index, top_k: int, cutoff: float):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    processor = SimilarityPostprocessor(similarity_cutoff=cutoff)
    return RetrieverQueryEngine(retriever=retriever, node_postprocessors=[processor])


def render_sources(source_nodes):
    if not source_nodes:
        st.info("No passages passed the similarity threshold.")
        return
    for i, node in enumerate(source_nodes, 1):
        page = node.node.metadata.get("source", node.node.metadata.get("page_label", "?"))
        score = f"{node.score:.2f}" if node.score is not None else "–"
        text = node.node.get_content().strip().replace("\n", " ")
        text = text[:420] + ("…" if len(text) > 420 else "")
        st.markdown(
            f"""<div class="source">
                <div class="meta">Passage {i} · page {page}<span class="badge">score {score}</span></div>
                {text}
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- session state
defaults = {"index": None, "doc_name": None, "n_pages": 0, "messages": [], "pending": None}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    api_key = st.text_input(
        "OpenRouter API key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Or set OPENROUTER_API_KEY in your .env file.",
    )
    llm_model = st.text_input("LLM model", value="openai/gpt-oss-20b")

    st.markdown("### 🔎 Retrieval")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 12, 4)
    cutoff = st.slider("Similarity cutoff", 0.0, 0.9, 0.40, 0.05)
    show_sources = st.toggle("Show source passages", value=True)

    st.divider()
    st.markdown("### 📤 Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded and st.button("Index document", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenRouter API key first.")
        else:
            configure_models(api_key, llm_model)
            with st.spinner("Chunking, embedding and indexing…"):
                st.session_state.index, st.session_state.n_pages = build_index(uploaded)
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
            st.success("Ready to chat!")

    if st.session_state.messages and st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- hero
st.markdown(
    """
<div class="hero">
    <h1>DocuMind</h1>
    <p>Upload a PDF and ask questions in plain language. Answers are grounded in your document, with the source passages shown.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- main
if st.session_state.index is None:
    st.markdown(
        """
<div class="empty">
    <div class="icon">📄</div>
    <h3>No document loaded</h3>
    <p>Add your API key and upload a PDF in the sidebar, then click <b>Index document</b>.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# stats row
c1, c2, c3 = st.columns(3)
for col, val, label in [
    (c1, st.session_state.doc_name, "Document"),
    (c2, st.session_state.n_pages, "Pages indexed"),
    (c3, len(st.session_state.messages) // 2, "Questions asked"),
]:
    col.markdown(
        f'<div class="stat"><div class="v">{val}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )
st.write("")

# suggested prompts when chat is empty
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    s1, s2, s3 = st.columns(3)
    suggestions = [
        "Summarize this document",
        "What are the key points?",
        "List the main conclusions",
    ]
    for col, q in zip((s1, s2, s3), suggestions):
        if col.button(q, use_container_width=True):
            st.session_state.pending = q
            st.rerun()

# chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources") is not None:
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                render_sources(msg["sources"])

# input
prompt = st.chat_input("Ask something about your document…")
if st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            configure_models(api_key, llm_model)
            engine = build_query_engine(st.session_state.index, top_k, cutoff)
            with st.spinner("Searching the document…"):
                res = engine.query(prompt)
            answer = res.response or ( # type: ignore
                "I couldn't find relevant information in the document. "
                "Try lowering the similarity cutoff or rephrasing your question."
            )
            sources = res.source_nodes
            st.markdown(answer)
            if show_sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
