import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def ingest_pdf(
    pdf_file,
    index_name: str,
    embeddings,
):
    """
    Ingest uploaded PDF into Pinecone
    """

    # 1. Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    try:
        # 2. Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # 3. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        docs = splitter.split_documents(pages)

        # 4. Add metadata
        enriched_docs = []
        for doc in docs:
            enriched_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "source": pdf_file.name,
                        "type": "pdf"
                    }
                )
            )

        # 5. Store in Pinecone
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )

        vectorstore.add_documents(enriched_docs)

        return len(enriched_docs)

    finally:
        os.remove(tmp_path)
