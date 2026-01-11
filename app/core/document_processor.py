import tempfile
from pathlib import Path
from typing import BinaryIO


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
)

from app.config import get_settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class DocumentProcessor:
    """Process documenets for the RAG pipeline."""

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".csv"}

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        """Initialize DocumentProcessor.

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        settings = get_settings()
        logger.info('Initializing DocumentProcessor...')
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.text_splitter = self._initialize_text_splitter()

        logger.info(
            f"DocumentProcessor initialized with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    
    def _initialize_text_splitter(self):
        """Initialize the text splitter with chunk size and overlap."""

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """Load and extract text from a PDF file.

        Args:
            file_path: Path to the PDF file
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        logger.info(f"Loading PDF document from: {file_path}")

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} pages from PDF: {file_path}")
        return documents


    def load_text(self, file_path: str | Path) -> list[Document]:
        """Load and extract text from a text file.

        Args:
            file_path: Path to the text file
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        logger.info(f"Loading text document from: {file_path}")

        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} documents from text file: {file_path}")
        return documents


    def load_csv(self, file_path: str | Path) -> list[Document]:
        """Load and extract text from a CSV file.

        Args:
            file_path: Path to the CSV file
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        logger.info(f"Loading CSV document from: {file_path}")

        loader = CSVLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} documents from CSV file: {file_path}")
        return documents


    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load and extract text from a file based on its extension.

        Args:
            file_path: Path to the file
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        loaders = {
            ".pdf": self.load_pdf,
            ".txt": self.load_text,
            ".csv": self.load_csv,
        }

        return loaders[extension](file_path)


    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks.

        Args:
            documents: List of Document objects
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Splitting {len(documents)} documents into chunks.")
        chunked_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunked_docs)} chunks from documents.")
        return chunked_docs


    def process_file(self, file_path: str | Path) -> list[Document]:
        """Load and process a file into chunked documents.

        Args:
            file_path: Path to the file
        Returns:
            List of chunked Document objects
        """
        documents = self.load_file(file_path)
        chunked_documents = self.split_documents(documents)
        return chunked_documents


    def load_from_uploaded_file(self, file: BinaryIO, filename: str) -> list[Document]:
        """Load and process an uploaded file into chunked documents.

        Args:
            file: Uploaded file object
            filename: Name of the uploaded file
        Returns:
            List of chunked Document objects
        """
        extension = Path(filename).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        # Save to temporary file on disk for processing
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        try:
            documents = self.load_file(tmp_path)

            # Update metadata with original filename
            for doc in documents:
                doc.metadata["source"] = filename

            return documents
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    

    def process_upload(self, file: BinaryIO, filename: str) -> list[Document]:
        """Load and process an uploaded file into chunked documents.

        Args:
            file: Uploaded file object
            filename: Name of the uploaded file
        Returns:
            List of chunked Document objects
        """
        documents = self.load_from_uploaded_file(file, filename)
        chunked_documents = self.split_documents(documents)
        return chunked_documents