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
