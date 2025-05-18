from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os

def get_reranker_model(
    model_name: str = "BAAI/bge-reranker-base",
    device: str = None
) -> HuggingFaceCrossEncoder:
    """
    Returns a HuggingFaceCrossEncoder model for reranking search results.
    Args:
        model_name: HuggingFace model name or path.
        device: Device to use ('cpu', 'cuda', etc). If None, uses EMBEDDING_DEVICE env var or defaults to 'cpu'.
    Returns:
        HuggingFaceCrossEncoder instance.
    """
    if device is None:
        device = os.environ.get("EMBEDDING_DEVICE", "cpu")
    print(f"[INFO] Loading reranker model '{model_name}' on device '{device}'")
    return HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs={"device": device}
    )
