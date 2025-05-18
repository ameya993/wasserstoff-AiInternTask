# Copyright (c) 2025 Ameya Gawande. All rights reserved.
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
#
# You may not use this file except in compliance with the License.
# Non-commercial use only. Commercial use is strictly prohibited without written permission.

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
