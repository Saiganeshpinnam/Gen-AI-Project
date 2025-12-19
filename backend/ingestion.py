from dotenv import load_dotenv
import os

from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

# Load env vars
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

print("DEBUG → INDEX NAME:", PINECONE_INDEX_NAME)

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Sample text
raw_text = """
Employees are allowed 12 paid leaves per year.
Work from home is allowed on Fridays.
Office timing is 9 AM to 6 PM.
"""

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_text(raw_text)

documents = [
    Document(
        page_content=chunk,
        metadata={"source": "hr_policy.txt", "department": "HR"}
    )
    for chunk in chunks
]

# ✅ FREE local embeddings (NO API KEY)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to existing index
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

# Add documents
vectorstore.add_documents(documents)

print("✅ Ingestion completed successfully (HuggingFace)")
