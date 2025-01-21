"""Database operations for document storage and retrieval."""
import logging
from typing import List, Optional, Dict
from langchain_openai import OpenAIEmbeddings
from langchain.schema.vectorstore import VectorStore
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class DocumentStore:
    """Handles document storage and retrieval using ChromaDB with LangChain integration."""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "ethics_docs",
        model: str = "text-embedding-3-small",
        dimensions: int = 1536
    ):
        """
        Initialize ChromaDB client and collection with LangChain embeddings.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to store documents
            model: OpenAI embedding model to use
            dimensions: Dimension of the embedding vectors
        """
        self._embedding_function = OpenAIEmbeddings(
            model=model,
            dimensions=dimensions
        )
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self._embedding_function,
            persist_directory=persist_directory
        )
    
    def add_documents(self, documents: List[dict], ids: Optional[List[str]] = None) -> None:
        """
        Add documents to the collection.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
            ids: Optional list of IDs for the documents
        """
        try:
            texts = [doc['content'] for doc in documents]
            metadatas = [doc.get('metadata', {}) for doc in documents]
            
            logger.info(f"Adding {len(documents)} documents to collection")
            self._vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the current collection."""
        try:
            logger.info(f"Deleting collection {self._collection_name}")
            self._vectorstore.delete_collection()
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def get_retriever(self, search_kwargs: Optional[Dict] = None) -> VectorStore:
        """
        Get the vector store retriever.
        
        Args:
            search_kwargs: Optional search parameters for the retriever
        """
        return self._vectorstore.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )