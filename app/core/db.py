"""Database operations for document storage and retrieval."""
import logging
from pathlib import Path
from typing import List, Optional, Dict

from langchain_ollama import OllamaEmbeddings
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class DocumentStore:
    """Handles document storage and retrieval using ChromaDB with LangChain integration."""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "docs",
        model: str = "llama3.1:8b",
        load_from_disk: bool = True
    ):
        """
        Initialize ChromaDB client - either load existing DB or create new.
        
        Args:
            persist_directory: Directory to persist/load the database
            collection_name: Name of the collection
            model: Ollama Embeddings model to use
            load_from_disk: If True and DB exists, load it; otherwise create new
        """
        self._embedding_function = OllamaEmbeddings(model=model)
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        db_exists = Path(persist_directory).exists()
        
        if db_exists and load_from_disk:
            logger.info(f"Loading existing Chroma DB from {persist_directory}")
            self._vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self._embedding_function,
                collection_name=collection_name
        )
        else:
            logger.info("Initializing new Chroma DB")
            self._vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self._embedding_function,
                persist_directory=persist_directory
    )
    
    def add_documents(self, documents: List[dict], ids: Optional[List[str]] = None) -> None:
        """
        Add documents to the collection.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys.
            ids: Optional list of IDs for the documents.
        """
        try:
            texts = [doc['content'] for doc in documents]
            metadatas = [doc.get('metadata', {}) for doc in documents]
            
            logger.info(f"Adding {len(documents)} documents to collection {self._collection_name}")
            self._vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the current collection entirely."""
        try:
            logger.info(f"Deleting collection {self._collection_name}")
            self._vectorstore.delete_collection()
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def get_retriever(self, search_kwargs: Optional[Dict] = None) -> VectorStoreRetriever:
        """
        Get the vector store retriever.
        
        Args:
            search_kwargs: Optional search parameters for the retriever.
        
        Returns:
            A VectorStore retriever configured with the desired search parameters.
        """
        return self._vectorstore.as_retriever(
            search_kwargs=search_kwargs or {"k": 3}
        )
