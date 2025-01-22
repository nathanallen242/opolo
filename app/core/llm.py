"""LLM operations for question answering using RAG."""
import logging
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.vectorstore import VectorStoreRetriever

logger = logging.getLogger(__name__)

class LLM:
    """Handles LLM operations with streamlined RAG pipeline."""
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )
    
    RAG_TEMPLATE = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    
    If the context doesn't contain enough information to answer the question fully, 
    say "I don't have enough information to answer that completely" and then provide 
    whatever partial information you can from the context."""
    
    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize LLM with the specified model.
        
        Args:
            model_name: Name of the Llama model to use
        """
        self._llm = ChatOllama(model=model_name, temperature=0.1)
        self._rag_prompt = ChatPromptTemplate.from_template(self.RAG_TEMPLATE)
    
    def _build_rag_chain(self, retriever: VectorStoreRetriever):
        """
        Build the RAG chain combining retrieval and response generation.
        
        Args:
            retriever: Vector store retriever to use
            
        Returns:
            Complete RAG chain
        """
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=self._llm,
            prompt=self.QUERY_PROMPT
        )
        
        return (
            {"context": mq_retriever, "question": RunnablePassthrough()}
            | self._rag_prompt
            | self._llm
            | StrOutputParser()
        )
    
    async def askQuestion(self, question: str, retriever: VectorStoreRetriever) -> str:
        """
        Process a question using the RAG pipeline.
        
        Args:
            question: The question to answer
            retriever: Vector store retriever
            
        Returns:
            Generated answer string
        """
        try:
            logger.info("Building RAG chain")
            chain = self._build_rag_chain(retriever)
            
            logger.info("Generating response")
            response = await chain.ainvoke(question)
            
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise