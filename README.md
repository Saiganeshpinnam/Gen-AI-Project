# 🤖 Enterprise Knowledge Assistant (RAG-based)

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions from internal enterprise documents and receive **accurate, contextual, and confidence-aware answers** with optional **audio playback**.

This project demonstrates a **production-style GenAI architecture** using modern AI tooling and cloud deployment.

---

## 🚀 Live Demo

Deployed on **Streamlit Cloud**  
👉 (Your deployed Streamlit URL goes here)

---

## 📌 Problem Statement

Enterprise knowledge is scattered across:
- PDFs
- HR policy documents
- Internal manuals and wikis

Employees spend significant time searching for answers or repeatedly asking support teams.

---

## 💡 Solution

This application uses **Retrieval-Augmented Generation (RAG)** to:
- Semantically search enterprise documents
- Retrieve only relevant content
- Generate grounded LLM responses
- Detect low-confidence answers
- Provide audio summaries for hands-free usage

---

## 🏗️ Architecture Overview

User Query
↓
HuggingFace Embeddings
↓
Pinecone Vector Database
↓
Relevant Context Retrieval
↓
Groq LLM (Llama 3)
↓
Agent Confidence Check
↓
Text + Audio Output (Streamlit UI)


---

## 🔑 Key Features

- ✅ Multi-format document ingestion (PDF, text)
- ✅ Semantic vector search
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Confidence-based agent logic
- ✅ Audio output (Text-to-Speech)
- ✅ Streamlit chat-style UI
- ✅ Cloud deployment with auto-redeploy

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### AI & Backend
- LangChain
- HuggingFace Embeddings
- Groq (Llama 3.1)
- Pinecone Vector Database

### Text-to-Speech
- Google Text-to-Speech (gTTS)

### Deployment
- Streamlit Cloud
- GitHub

---

## 📂 Project Structure

rag-project/
│
├── backend/
│ ├── app.py # Streamlit UI
│ ├── rag.py # RAG + Agent logic
│ ├── ingestion.py # Text ingestion
│ ├── pdf_ingestion.py # PDF ingestion
│ ├── tts.py # Text-to-Speech
│ ├── requirements.txt
│ ├── Procfile
│ ├── runtime.txt
│
├── .gitignore
├── README.md


---

## ⚙️ Environment Variables

Set the following variables in **Streamlit Cloud → App Settings → Secrets**:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-index
GROQ_API_KEY=your_groq_api_key

❗ Do NOT commit .env files to GitHub.

📥 Document Ingestion

Run ingestion locally before deployment:

python ingestion.py
python pdf_ingestion.py


This will:

Chunk documents

Generate embeddings

Store vectors in Pinecone

🔎 Retrieval & Answer Generation (RAG)

User query → embeddings

Retrieve top-k relevant chunks from Pinecone

Provide context to LLM

Generate grounded answer

Evaluate confidence score

🧠 Agent Layer Logic
Condition	Action
High similarity score	Answer user
Low similarity score	Escalate / fallback
No context	Respond with insufficient information

🔊 Audio Output

Generated answers are converted to speech

Audio is playable directly in the UI

🚀 Deployment (Streamlit Cloud)

Push code to GitHub

Create app on Streamlit Cloud

Set Main file path:

backend/app.py


Add environment secrets

App auto-redeploys on every GitHub push

🔊 Audio Output

Generated answers are converted to speech

Audio is playable directly in the UI

🚀 Deployment (Streamlit Cloud)

Push code to GitHub

Create app on Streamlit Cloud

Set Main file path:

backend/app.py


Add environment secrets

App auto-redeploys on every GitHub push

📈 Future Enhancements

Slack / Email escalation

User authentication

Analytics dashboard

Role-based document access

Multi-language support

Streaming responses

🧑‍💼 Interview Highlights

Real-world enterprise RAG implementation

Clear separation of layers

Agent-based decision making

Cloud deployment

Scalable architecture








