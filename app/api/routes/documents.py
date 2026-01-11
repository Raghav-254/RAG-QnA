"""Document management endpoints."""

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.api.schemas import (
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse
)
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger
from app.core.document_processor import DocumentProcessor

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
)
async def upload_document(file: UploadFile = File(..., description="Document file to upload")) -> DocumentUploadResponse:
    """Upload a document and process it into chunks.

    Args:
        file: Uploaded document file

    Returns:
        DocumentUploadResponse: Details about the uploaded document and processing status
    """
    logger.info(f"Received document upload: {file.filename}")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    try: 
        # Process uploaded file into chunked documents
        processor = DocumentProcessor()
        chunked_docs = processor.load_from_uploaded_file(file.file, file.filename)
        vector_store = VectorStoreService()

        # Add chunked documents to vector store
        document_ids = vector_store.add_documents(chunked_docs)
        logger.info(
            f"Successfully processed {file.filename}: "
            f"{len(chunked_docs)} chunks, {len(document_ids)} documents"
        )
        response = DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            chunks_created=len(chunked_docs),
            document_ids=document_ids
        )
        return response
    except ValueError as ve:
        logger.error(f"Upload error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Error processing document")


@router.get(
    "/info",
    response_model=DocumentListResponse,
    summary="Get collection information",
    description="Get information about the document collection.",
)
async def get_collection_info() -> DocumentListResponse:
    """Get information about the document collection."""
    logger.debug("Collection info requested")

    try:
        vector_store = VectorStoreService()
        info = vector_store.get_collection_info()

        return DocumentListResponse(
            collection_name=info["name"],
            total_documents=info["points_count"],
            status=info["status"],
        )
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting collection info: {str(e)}",
        )


@router.delete(
    "/collection",
    responses={
        200: {"description": "Collection deleted successfully"},
        500: {"model": ErrorResponse, "description": "Deletion error"},
    },
    summary="Delete the entire collection",
    description="Delete all documents from the vector store. Use with caution!",
)
async def delete_collection() -> dict:
    """Delete the entire document collection."""
    logger.warning("Collection deletion requested")

    try:
        vector_store = VectorStoreService()
        vector_store.delete_collection()

        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection: {str(e)}",
        )