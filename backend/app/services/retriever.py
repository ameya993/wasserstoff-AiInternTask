# Copyright (c) 2025 Ameya Gawande. All rights reserved.
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
#
# You may not use this file except in compliance with the License.
# Non-commercial use only. Commercial use is strictly prohibited without written permission.

from langchain_community.vectorstores import FAISS

def build_vectorstore(documents, embedding_model):
    """
    Build a FAISS vectorstore from a list of documents and an embedding model.

    Args:
        documents: List of document chunks to index.
        embedding_model: An embedding model instance.

    Returns:
        FAISS vectorstore object.
    """
    print(f"[INFO] Building FAISS vectorstore with {len(documents)} documents...")
    return FAISS.from_documents(documents=documents, embedding=embedding_model)
