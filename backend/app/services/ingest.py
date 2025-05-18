# Copyright (c) 2025 Ameya Gawande. All rights reserved.
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
#
# You may not use this file except in compliance with the License.
# Non-commercial use only. Commercial use is strictly prohibited without written permission.




import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.embedding import get_embedding_model


VECTOR_STORE_DIR = "data/faiss_index"

def build_faiss_index(pdf_paths):
    """
    Ingest one or more PDFs into the FAISS vectorstore.
    """
    embedding_model = get_embedding_model()
    all_texts = []
    for pdf_path in pdf_paths:
        try:
            print(f"Loading document from: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents. Splitting...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = splitter.split_documents(documents)
            print(f"Split into {len(texts)} chunks.")
            all_texts.extend(texts)
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path}: {e}")

    if not all_texts:
        print("[ERROR] No text chunks to index. Exiting.")
        return

    print(f"Embedding {len(all_texts)} chunks...")
    # If FAISS index exists, load and add; else, create new
    if os.path.exists(VECTOR_STORE_DIR):
        print("Existing FAISS index found. Loading and adding documents...")
        vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embedding_model, allow_dangerous_deserialization=True)
        vectorstore.add_documents(all_texts)
    else:
        print("No existing FAISS index found. Creating new index...")
        vectorstore = FAISS.from_documents(all_texts, embedding_model)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_DIR)
    print(f"✅ FAISS index saved at: {VECTOR_STORE_DIR}")
    print(f"Total chunks indexed: {len(all_texts)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf> [<path_to_pdf2> ...]")
        sys.exit(1)
    build_faiss_index(sys.argv[1:])
