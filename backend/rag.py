from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not set")

if not PINECONE_INDEX_NAME:
    raise ValueError("❌ PINECONE_INDEX_NAME not set")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set")

# 🔥 IMPORTANT: Pinecone distance threshold (lower = better)
CONFIDENCE_THRESHOLD = 0.6

print(f"✅ Using Pinecone Index: {PINECONE_INDEX_NAME}")

# ----------------------------
# Pinecone
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# ----------------------------
# Embeddings (MUST match ingestion)
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# Vector Store
# ----------------------------
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)

# ----------------------------
# LLM (Groq)
# ----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# ----------------------------
# Prompt Template
# ----------------------------
PROMPT_TEMPLATE = """
You are an enterprise knowledge assistant.

Answer the question using ONLY the context below.
If the answer is not present, say:
"I don't have enough information to answer this."

Context:
{context}

Question:
{question}

Answer:
"""

# ----------------------------
# RAG Pipeline
# ----------------------------
def generate_answer(query: str):
    # 1️⃣ Retrieve documents + distance scores
    results = vectorstore.similarity_search_with_score(query, k=3)

    if not results:
        return {
            "answer": "I don't have enough information to answer this.",
            "confidence": 0.0,
            "action": "fallback"
        }

    docs = []
    scores = []

    for doc, score in results:
        docs.append(doc)
        scores.append(score)

    # 2️⃣ Pinecone distance logic
    best_score = min(scores)          # lower = better
    confidence = round(1 - best_score, 2)  # human-friendly confidence

    # 3️⃣ Agent decision
    if best_score > CONFIDENCE_THRESHOLD:
        return {
            "answer": "I'm not confident enough to answer this. Escalating.",
            "confidence": confidence,
            "action": "escalate"
        }

    # 4️⃣ Build context
    context = "\n\n".join(
        f"{doc.page_content}\n(Source: {doc.metadata})"
        for doc in docs
    )

    # 5️⃣ Prompt LLM
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    ).format(
        context=context,
        question=query
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "confidence": confidence,
        "action": "answered"
    }
