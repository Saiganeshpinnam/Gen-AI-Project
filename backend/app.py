from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag import generate_answer
from tts import text_to_speech

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

# ----------------------------
# Title
# ----------------------------
st.title("🤖 Enterprise Knowledge Assistant")
st.write("Ask questions from internal company documents")

# ----------------------------
# Session state for chat
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# User Input
# ----------------------------
user_query = st.text_input(
    "Enter your question:",
    placeholder="e.g. How many paid leaves are employees allowed?"
)

# ----------------------------
# Ask Button
# ----------------------------
if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            result = generate_answer(user_query)

        # 🔊 Convert answer to speech
        audio_path = text_to_speech(result["answer"])

        # Save chat history (include audio)
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

    # 🔊 Audio playback
    if chat.get("audio"):
        st.audio(chat["audio"], format="audio/mp3")

    st.markdown(
        f"""
        **Confidence:** `{round(chat['confidence'], 2)}`  
        **Agent Action:** `{chat['action']}`
        """
    )

    st.divider()
