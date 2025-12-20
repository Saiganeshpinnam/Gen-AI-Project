from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os

from rag import generate_answer
from tts import text_to_speech
from pdf_ingestion import ingest_pdf

from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Config
# ----------------------------
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise Knowledge Assistant")
st.write("Ask questions from internal company documents")

# ----------------------------
# Embeddings (same everywhere)
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# PDF Upload Section
# ----------------------------
st.subheader("📄 Upload Documents")

uploaded_pdf = st.file_uploader(
    "Upload a PDF file",
    type=["pdf", "docx", "doc"]

)

if uploaded_pdf:
    if st.button("📥 Ingest PDF"):
        with st.spinner("Processing and indexing PDF..."):
            chunk_count = ingest_pdf(
                pdf_file=uploaded_pdf,
                index_name=PINECONE_INDEX_NAME,
                embeddings=embeddings
            )

        st.success(f"✅ PDF ingested successfully ({chunk_count} chunks added)")

st.divider()

# ----------------------------
# Session state
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# User Query
# ----------------------------
user_query = st.text_input(
    "Enter your question:",
    placeholder="e.g. What is the leave policy?"
)

if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            result = generate_answer(user_query)

        audio_path = text_to_speech(result["answer"])

        st.session_state.chat_history.append(
            {
                "question": user_query,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "action": result["action"],
                "audio": audio_path
            }
        )
    else:
        st.warning("Please enter a question.")

# ----------------------------
# Display Chat History
# ----------------------------
for chat in reversed(st.session_state.chat_history):
    st.markdown("### 🧑 User")
    st.write(chat["question"])

    st.markdown("### 🤖 Assistant")
    st.write(chat["answer"])

    if chat.get("audio"):
        st.audio(chat["audio"], format="audio/mp3")

    st.markdown(
        f"""
        **Confidence:** `{round(chat['confidence'], 2)}`  
        **Agent Action:** `{chat['action']}`
        """
    )

    st.divider()
