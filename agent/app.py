import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key from .env
load_dotenv()
jina_api_key = os.getenv("JINA_API_KEY")

# 1. Load the PDF document
def load_pdf(path):
    loader = PyMuPDFLoader(path)
    return loader.load()

# 2. Split document into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

# 3. Generate embeddings using Jina
def embed_documents(documents):
    embedding_model = JinaEmbeddings(
        jina_api_key=jina_api_key,
        model_name="jina-embeddings-v2-base-en"
    )
    return FAISS.from_documents(documents, embedding_model)

# 4. Save vector store locally
def save_vector_store(vectorstore, folder="vectorstore"):
    vectorstore.save_local(folder)

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"  # 👈 Place your PDF in /data with this name

    print("📄 Loading PDF...")
    docs = load_pdf(pdf_path)

    print(f"✂️ Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"🔹 Total chunks: {len(chunks)}")

    print("🔍 Generating embeddings with Jina...")
    vectorstore = embed_documents(chunks)

    print("💾 Saving FAISS vector store...")
    save_vector_store(vectorstore)

    print("✅ Vector store ready! You can now query it using a local LLM.")
