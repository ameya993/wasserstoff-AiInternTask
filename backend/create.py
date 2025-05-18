import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.embedding import get_embedding_model  # adjust import as needed

VECTOR_STORE_PATH = "data/faiss_index"
PDF_PATH = r"Wasserstoff Gen-AI Internship Task.pdf"

# 1. Load and split the document
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# 2. Get embedding model
embedding_model = get_embedding_model()

# 3. Build FAISS index from chunks
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 4. Ensure directory exists, then save
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
vectorstore.save_local(VECTOR_STORE_PATH)

print("✅ index.faiss created at:", VECTOR_STORE_PATH)
