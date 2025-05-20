# Copyright (c) 2025 Ameya Gawande. All rights reserved.
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
#
# You may not use this file except in compliance with the License.
# Non-commercial use only. Commercial use is strictly prohibited without written permission.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ".", " "],  # Try to split by paragraphs, then sentences, then words
    )
    # Optionally, add paragraph number to metadata for citation
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(splits):
            meta = dict(doc.metadata)
            meta["paragraph"] = i + 1
            chunks.append(Document(page_content=chunk, metadata=meta))
    return chunks
