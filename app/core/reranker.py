from sentence_transformers import CrossEncoder
from typing import List
from langchain.schema import Document

class Reranker:
    """Cross-encoder reranker using sentence-transformers."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, max_length=512)
        
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query."""
        if not documents:
            return documents
            
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs]