# Copyright (c) 2025 Ameya Gawande. All rights reserved.
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
#
# You may not use this file except in compliance with the License.
# Non-commercial use only. Commercial use is strictly prohibited without written permission.




from pydantic import BaseSettings

class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    DEVICE: str = "cpu" 
    VECTOR_DB_PATH: str = "vector_db"
    UPLOAD_FOLDER: str = "uploaded_docs"

settings = Settings()
