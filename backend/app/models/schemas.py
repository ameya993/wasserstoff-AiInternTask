from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""
    document_id: str = Field(..., description="Unique ID for the uploaded document")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Upload status message")

class QueryRequest(BaseModel):
    """Request model for querying documents."""
    query: str = Field(..., description="The research question or query")
    selected_docs: Optional[List[str]] = Field(None, description="Optional list of document IDs to restrict the query")

class DocumentAnswer(BaseModel):
    """Answer to a query for a specific document."""
    document_id: str = Field(..., description="ID of the document")
    answer: str = Field(..., description="Answer extracted from the document")
    citation: str = Field(..., description="Citation for the answer")

class ThemeResponse(BaseModel):
    """Response model for theme identification and synthesis."""
    themes: List[str] = Field(..., description="List of synthesized themes")
    theme_citations: List[List[str]] = Field(..., description="List of document IDs for each theme")
    synthesized_answer: str = Field(..., description="Synthesized answer across documents")
