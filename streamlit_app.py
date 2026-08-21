"""Streamlit UI for the Document Summarization and Q&A Service."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st


from doc_summarizer.config.settings import Settings, get_settings
from doc_summarizer.ingestion.factory import LoaderFactory
from doc_summarizer.summarization.factory import SummarizerFactory


# ---- Page configuration ----
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="D",
    layout="centered",
)


# ---- Cached resources (load models once) ----
@st.cache_resource
def load_summarizer(quantize: bool):
    """Load and cache the summarizer."""
    base = get_settings()
    settings_dict = base.model_dump()
    settings_dict["quantize"] = quantize
    fresh = Settings(**settings_dict)
    return SummarizerFactory.create(fresh)


@st.cache_resource
def load_qa_agent():
    """Load and cache the generative Q&A agent."""
    from doc_summarizer.agent.generative_qa import GenerativeQAAgent

    return GenerativeQAAgent()

# ---- Helper: save upload to a temp file ----
def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


# ---- Header ----
st.title("Document Intelligence Service")
st.caption("Upload a document to get a summary and ask questions about it.")

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    st.markdown("Q&A powered by a local LLM (Ollama).")
    st.divider()
    st.markdown("**Supported formats:**")
    st.markdown(", ".join(LoaderFactory.supported_extensions()))


# ---- File uploader ----
uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "txt", "docx"],
)

# ---- Track the current document in session state ----
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "indexed_doc" not in st.session_state:
    st.session_state.indexed_doc = None


if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")

    # ---- SUMMARIZATION SECTION ----
    st.subheader("Summary")
    if st.button("Summarize", type="primary"):
        temp_path = save_upload(uploaded_file)
        try:
            with st.spinner("Summarizing..."):
                document = LoaderFactory.get_loader(temp_path).load()
                summarizer = load_summarizer(False)
                start = time.perf_counter()
                summary = summarizer.summarize(document.text)
                elapsed = time.perf_counter() - start

            st.write(summary)

            original_chars = len(document.text)
            summary_chars = len(summary)
            ratio = round(summary_chars / original_chars, 3) if original_chars else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Original", f"{original_chars} chars")
            col2.metric("Summary", f"{summary_chars} chars")
            col3.metric("Compression", f"{ratio}")
            st.caption(f"Processed in {elapsed:.2f} seconds.")

            st.download_button(
                label="Download summary",
                data=summary,
                file_name=f"{uploaded_file.name}.summary.txt",
                mime="text/plain",
            )
        except Exception as exc:
            st.error(f"Failed to summarize: {exc}")
        finally:
            temp_path.unlink(missing_ok=True)

    # ---- Q&A SECTION ----
    st.divider()
    st.subheader("Ask Questions")

    question = st.text_input(
        "Ask a question about the document:",
        placeholder="e.g., What is the main topic?",
    )

    if question:
        temp_path = save_upload(uploaded_file)
        try:
            with st.spinner("Answering..."):
                agent = load_qa_agent()

                doc_key = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get("indexed_doc") != doc_key:
                    agent.index_document(temp_path)
                    st.session_state.indexed_doc = doc_key

                result = agent.ask(question)

            st.markdown(f"**Answer:** {result['answer']}")

            confidence = result.get("confidence", 0)
            if confidence >= 0.6:
                st.success(f"Confidence: {confidence}")
            else:
                st.warning(
                    f"Confidence: {confidence} — "
                    "the document may not fully answer this question."
                )

            with st.expander("Show context used"):
                st.caption(result["context_used"])
        except Exception as exc:
            st.error(f"Failed to answer: {exc}")
        finally:
            temp_path.unlink(missing_ok=True)