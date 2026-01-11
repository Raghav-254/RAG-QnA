"""
Pydantic schemas for API request and response models
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ============== Health Schemas ==============
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )
    version: str = Field(..., description="Application version")


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    status: str = Field(..., description="Service status")
    qdrant_connected: bool = Field(..., description="Qdrant connection status")
    collection_info: dict = Field(..., description="Collection information")


# ============== Document Schemas ==============
class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    message: str = Field(..., description="Upload status message")
    filename: str = Field(..., description="Name of the uploaded file")
    chunks_created: int = Field(..., description="Number of chunks created from the document")
    document_ids: list[str] = Field(..., description="List of document IDs created")


class DocumentInfo(BaseModel):
    """Document information."""
    source: str = Field(..., description="Source of the document")
    metadata: dict[str, Any] = Field(..., description="Metadata associated with the document")


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    collection_name: str = Field(..., description="Name of the collection")
    total_documents: int = Field(..., description="Total number of documents in the collection")
    status: str = Field(..., description="Collection status")


# ============== Query Schemas ==============
class QueryRequest(BaseModel):
    """Request for RAG query."""

    question: str = Field(
        ...,
        description="Question to ask",
        min_length=1,
        max_length=1000,
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response",
    )
    enable_evaluation: bool = Field(
        default=False,
        description="Enable RAGAS evaluation (faithfulness, answer relevancy)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is RAG?",
                    "include_sources": True,
                    "enable_evaluation": False,
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    """Source document information."""

    content: str = Field(..., description="Document content excerpt")
    metadata: dict[str, Any] = Field(..., description="Document metadata")


class EvaluationScores(BaseModel):
    """RAGAS evaluation scores."""

    faithfulness: float | None = Field(
        None,
        description="Faithfulness score (0-1): measures factual consistency with sources",
        ge=0.0,
        le=1.0,
    )
    answer_relevancy: float | None = Field(
        None,
        description="Answer relevancy score (0-1): measures relevance to question",
        ge=0.0,
        le=1.0,
    )
    evaluation_time_ms: float | None = Field(
        None,
        description="Time taken for evaluation in milliseconds",
    )
    error: str | None = Field(
        None,
        description="Error message if evaluation failed",
    )


class QueryResponse(BaseModel):
    """Response for RAG query."""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDocument] | None = Field(
        None,
        description="Source documents used",
    )
    processing_time_ms: float = Field(
        ...,
        description="Query processing time in milliseconds",
    )
    evaluation: EvaluationScores | None = Field(
        None,
        description="RAGAS evaluation scores (if requested)",
    )


# ============== Error Schemas ==============
class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    error: str = Field(default="Validation Error", description="Error type")
    message: str = Field(..., description="Error message")
    errors: list[dict] = Field(..., description="Validation errors")