"""Embedding generation module using OpenAI embeddings"""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

@lru_cache
def get_embedding_model() -> OpenAIEmbeddings:
    """Get the OpenAI embedding model instance.

    Returns:
        OpenAIEmbeddings: Instance of the OpenAI embeddings model
    """
    settings = get_settings()
    logger.info(f"Initializing OpenAI Embedding Model: {settings.embedding_model}")
    embedding_model = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return embedding_model

class EmbeddingService:
    """Service for generating embeddings."""

    def __init__(self) -> None:
        """Initialize the EmbeddingService with cached embedding model."""
        self.embedding_model = get_embedding_model()
        logger.info("EmbeddingService initialized with OpenAI Embedding Model.")

    def generate_query_embedding(self, text: str) -> list[float]:
        """Generate embedding for the given text.

        Args:
            text: Input text to generate embedding for

        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.embedding_model.embed_query(text)
        logger.debug(f"Generated embedding of length {len(embedding)} for text.")
        return embedding

    def generate_document_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of input texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.embed_documents(texts)
        logger.debug(f"Generated embeddings for {len(embeddings)} texts.")
        return embeddings