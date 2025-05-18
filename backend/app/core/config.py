from pydantic import BaseSettings

class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    DEVICE: str = "cpu" 
    VECTOR_DB_PATH: str = "vector_db"
    UPLOAD_FOLDER: str = "uploaded_docs"

settings = Settings()
