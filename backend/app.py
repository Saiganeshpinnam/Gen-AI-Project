from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

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
# Embeddings
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# Upload & Ingestion
# ----------------------------
st.subheader("📄 Upload Documents")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx"]
)

if uploaded_file:
    if st.button("📥 Ingest Document"):
        with st.spinner("Processing and indexing document..."):
            chunk_count = ingest_pdf(
                uploaded_file,
                PINECONE_INDEX_NAME,
                embeddings
            )

        st.success(f"✅ Document ingested successfully ({chunk_count} chunks added)")

st.divider()

# ----------------------------
# Session State
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# Query Input
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

        st.session_state.chat_history.append({
            "question": user_query,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "action": result["action"],
            "audio": audio_path
        })
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
