from dotenv import load_dotenv
import os

# ----------------------------
# LangChain + Vector DB
# ----------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ----------------------------
# Groq LLM
# ----------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Config
# ----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CONFIDENCE_THRESHOLD = 0.75  # 👈 ADD THIS LINE HERE

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not set in .env")

if not PINECONE_INDEX_NAME:
    raise ValueError("❌ PINECONE_INDEX_NAME not set in .env")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set in .env")

print(f"✅ Using Pinecone Index: {PINECONE_INDEX_NAME}")

# ----------------------------
# Initialize Pinecone
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# ----------------------------
# Embedding Model (MUST match ingestion)
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# Connect to Vector Store
# ----------------------------
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)

# ----------------------------
# Retrieval Function
# ----------------------------
def retrieve_documents(query: str, k: int = 3):
    """
    Retrieve top-k documents WITH similarity scores
    """
    if not query.strip():
        return []

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k
    )
    return results


# ----------------------------
# Initialize Groq LLM (Llama 3)
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

Answer the question using ONLY the context provided below.
If the answer is not present, say:
"I don't have enough information to answer this."

Context:
{context}

Question:
{question}

Answer:
"""

# ----------------------------
# RAG Answer Generator
# ----------------------------
def generate_answer(query: str):
    # 1. Retrieve documents with scores
    results = retrieve_documents(query, k=3)

    if not results:
        return {
            "answer": "I don't have enough information to answer this.",
            "confidence": 0.0,
            "action": "fallback"
        }

    # 2. Extract docs + scores
    docs = []
    scores = []

    for doc, score in results:
        docs.append(doc)
        scores.append(score)

    max_confidence = max(scores)

    # 3. Low confidence → agent fallback
    if max_confidence < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I'm not confident enough to answer this. I'll escalate this query.",
            "confidence": max_confidence,
            "action": "escalate"
        }

    # 4. Build context
    context = "\n\n".join(
        f"{doc.page_content}\n(Source: {doc.metadata})"
        for doc in docs
    )

    # 5. Prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    ).format(
        context=context,
        question=query
    )

    # 6. Call LLM
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "confidence": max_confidence,
        "action": "answered"
    }


# ----------------------------
# Run Test
# ----------------------------
if __name__ == "__main__":
    query = "How many paid leaves are employees allowed?"

    result = generate_answer(query)

    print("\n🤖 Agent Decision:\n")
    print("Answer:", result["answer"])
    print("Confidence:", round(result["confidence"], 2))
    print("Action:", result["action"])

