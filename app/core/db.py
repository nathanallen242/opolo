"""Database operations for document storage and retrieval."""
import logging
from typing import List, Optional, Dict

from langchain_ollama import OllamaEmbeddings
from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class DocumentStore:
    """Handles document storage and retrieval using ChromaDB with LangChain integration."""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "ethics_docs",
        model: str = "llama3.2"
    ):
        """
        Initialize ChromaDB client and collection with Ollama embeddings (Llama 2).
        
        Args:
            persist_directory: Directory to persist the database.
            collection_name: Name of the collection to store documents.
            model: Ollama Embeddings model to use (default is "llama2").
        """
        self._embedding_function = OllamaEmbeddings(model=model)
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        
        # Chroma vector store using Ollama embeddings
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
            search_kwargs=search_kwargs or {"k": 4}
        )
