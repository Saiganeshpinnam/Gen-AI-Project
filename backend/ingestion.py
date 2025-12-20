import os
import tempfile
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not set")


def ingest_pdf(uploaded_file, index_name: str, embeddings) -> int:
    """
    Ingest PDF / DOC / DOCX into Pinecone
    Returns number of chunks ingested
    """

    # 1️⃣ Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        # 2️⃣ Choose loader based on file type
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif file_name.endswith(".doc") or file_name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(temp_path)
        else:
            raise ValueError("❌ Unsupported file type")

        # 3️⃣ Load document pages
        pages = loader.load()

        if not pages:
            raise ValueError("❌ No content found in document")

        # 4️⃣ Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(pages)

        # 5️⃣ Add metadata
        documents = [
            Document(
                page_content=chunk.page_content,
                metadata={
                    **chunk.metadata,
                    "source": uploaded_file.name
                }
            )
            for chunk in chunks
        ]

        # 6️⃣ Store in Pinecone
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        vectorstore.add_documents(documents)

        return len(documents)

    finally:
        # 7️⃣ Cleanup temp file
        os.remove(temp_path)
