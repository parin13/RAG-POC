#!/usr/bin/env python3
"""
A Retrieval-Augmented Generation (RAG) system using LlamaIndex and Qdrant vector store.
Loads documents from a directory, indexes them into Qdrant, and queries the index for responses.
All data is stored solely in Qdrant, with no disk-based persistence.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost:6333")
COLLECTION_NAME = "first_test_pdf_rag"
DATA_DIR = Path("pdfData")

def initialize_vector_store() -> QdrantVectorStore:
    """Initializes and returns the Qdrant vector store."""
    try:
        qdrant_client = QdrantClient(QDRANT_HOST)
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME
        )
        logger.info(f"Connected to Qdrant at {QDRANT_HOST} with collection {COLLECTION_NAME}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant at {QDRANT_HOST}: {e}")
        raise RuntimeError(f"Qdrant initialization failed: {e}")

def check_collection_exists(qdrant_client: QdrantClient) -> bool:
    """Checks if the specified collection exists in Qdrant."""
    try:
        exists = qdrant_client.collection_exists(COLLECTION_NAME)
        return exists
    except UnexpectedResponse as e:
        logger.error(f"Error checking collection existence: {e}")
        raise

def load_documents(data_dir: Path) -> list:
    """Loads documents from the specified directory."""
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Data directory {data_dir} does not exist or is not a directory")
    try:
        reader = SimpleDirectoryReader(str(data_dir))
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents from {data_dir}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents from {data_dir}: {e}")
        raise

def create_or_load_index() -> VectorStoreIndex:
    """Creates a new index or loads an existing one from Qdrant."""
    try:
        # Initialize Qdrant client and vector store
        vector_store = initialize_vector_store()
        qdrant_client = QdrantClient(QDRANT_HOST)
        
        # Check if collection exists
        if not check_collection_exists(qdrant_client):
            logger.info("Collection does not exist, creating new index...")
            documents = load_documents(DATA_DIR)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Explicitly writing embeddings to Qdrant
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            logger.info(f"Vector embeddings written to Qdrant collection {COLLECTION_NAME}")
        else:
            logger.info("Collection exists, loading index from Qdrant...")
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            logger.info("Index loaded from Qdrant")
        return index
    except Exception as e:
        logger.error(f"Error creating or loading index: {e}")
        raise

def main(query: str):
    """Main entry point for the RAG system."""
    # Set and validate OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise ValueError("OPENAI_API_KEY is required")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    try:
        # Create or load the index
        index = create_or_load_index()
        
        # Query the index
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        logger.info(f"Query response: {response}")
        print(response)
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main("write a poem about the author")