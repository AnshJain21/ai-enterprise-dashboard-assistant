"""
AI Enterprise Dashboard Assistant
==================================
Free-tier stack: Streamlit + Groq API (free) + local embeddings + ChromaDB (local, free).

Run locally:
    1. pip install -r requirements.txt
    2. cp .env.example .env   and add your free Groq API key
    3. streamlit run app.py
"""
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.data_qa import answer_question
from utils.doc_qa import get_collection, add_document, ask, summarize_document

load_dotenv()

st.set_page_config(page_title="AI Enterprise Dashboard Assistant", layout="wide")

# ---------- session state ----------
if "chroma_collection" not in st.session_state:
    st.session_state.chroma_collection = get_collection()
if "uploaded_doc_names" not in st.session_state:
    st.session_state.uploaded_doc_names = []
if "data_chat_history" not in st.session_state:
    st.session_state.data_chat_history = []
if "doc_chat_history" not in st.session_state:
    st.session_state.doc_chat_history = []

# ---------- sidebar: API key check ----------
st.sidebar.title("⚙️ Setup")
api_key_present = bool(os.environ.get("GROQ_API_KEY"))
if api_key_present:
    st.sidebar.success("Groq API key detected")
else:
    st.sidebar.error("No GROQ_API_KEY found")
    st.sidebar.markdown(
        "Get a free key at [Groq Console](https://console.groq.com/keys), "
        "then add it to a `.env` file (see `.env.example`)."
    )

st.title("📊 AI Enterprise Dashboard Assistant")
st.caption("Ask questions about your data and summarize documents — powered by a free Groq API tier.")

tab_data, tab_docs = st.tabs(["📈 Data Q&A", "📄 Document Summarization"])

# =========================================================
# TAB 1: Data Q&A
# =========================================================
with tab_data:
    st.subheader("Upload structured data (CSV / Excel)")
    data_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"], key="data_upload")

    if data_file is not None:
        try:
            if data_file.name.endswith(".csv"):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)
            st.session_state.df = df
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't read file: {e}")

    if "df" in st.session_state:
        df = st.session_state.df
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"{len(df)} rows × {len(df.columns)} columns loaded")

        st.divider()
        st.subheader("Ask a question about this data")
        question = st.text_input(
            "e.g. 'What were total sales by region?' or 'Which product had the highest growth?'",
            key="data_question",
        )
        if st.button("Ask", key="data_ask_btn") and question:
            if not api_key_present:
                st.warning("Add your Groq API key first (see sidebar).")
            else:
                with st.spinner("Thinking..."):
                    result = answer_question(question, df)
                st.session_state.data_chat_history.append((question, result))

        for q, r in reversed(st.session_state.data_chat_history):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(r["answer"])
                with st.expander("Show generated pandas expression"):
                    st.code(r["expression"], language="python")
    else:
        st.info("Upload a CSV or Excel file to get started.")

# =========================================================
# TAB 2: Document Summarization / Q&A
# =========================================================
with tab_docs:
    st.subheader("Upload documents (PDF / DOCX / TXT)")
    doc_files = st.file_uploader(
        "Upload one or more documents", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="doc_upload"
    )

    if doc_files:
        for f in doc_files:
            if f.name not in st.session_state.uploaded_doc_names:
                if not api_key_present:
                    st.warning("Add your Groq API key first (see sidebar).")
                    break
                with st.spinner(f"Indexing {f.name}..."):
                    n_chunks = add_document(st.session_state.chroma_collection, f.name, f.read())
                st.session_state.uploaded_doc_names.append(f.name)
                st.success(f"Indexed {f.name} ({n_chunks} chunks)")

    if st.session_state.uploaded_doc_names:
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Summarize a document")
            doc_to_summarize = st.selectbox("Choose a document", st.session_state.uploaded_doc_names)
            if st.button("Summarize", key="summarize_btn"):
                with st.spinner("Summarizing..."):
                    summary = summarize_document(st.session_state.chroma_collection, doc_to_summarize)
                st.markdown(summary)

        with col2:
            st.subheader("Ask questions across all documents")
            doc_question = st.text_input("e.g. 'What risks were flagged in the Q3 report?'", key="doc_question")
            if st.button("Ask", key="doc_ask_btn") and doc_question:
                with st.spinner("Searching documents..."):
                    result = ask(st.session_state.chroma_collection, doc_question)
                st.session_state.doc_chat_history.append((doc_question, result))

            for q, r in reversed(st.session_state.doc_chat_history):
                with st.chat_message("user"):
                    st.write(q)
                with st.chat_message("assistant"):
                    st.write(r["answer"])
                    if r["sources"]:
                        st.caption(f"Sources: {', '.join(r['sources'])}")
    else:
        st.info("Upload at least one document to summarize or ask questions about it.")
