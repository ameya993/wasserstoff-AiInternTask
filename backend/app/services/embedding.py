from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os

def get_embedding_model(
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = None
) -> HuggingFaceBgeEmbeddings:
    """
    Returns a HuggingFaceBgeEmbeddings model for use in vectorstores.
    Args:
        model_name: HuggingFace model name or path.
        device: Device to use ('cpu', 'cuda', etc). If None, uses env var or defaults to 'cpu'.
    Returns:
        HuggingFaceBgeEmbeddings instance.
    """
    if device is None:
        device = os.environ.get("EMBEDDING_DEVICE", "cpu")
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    print(f"[INFO] Loading embedding model '{model_name}' on device '{device}'")
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
