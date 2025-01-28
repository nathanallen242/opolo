"""LLM operations for question answering using RAG."""
import logging
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.schema import BaseChatMessageHistory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
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
    
    RAG_TEMPLATE = """Here is the conversation so far:
    {history}

    Use ONLY the following context to answer:
    {context}
    Question: {question}

    If the context doesn't contain enough information to answer fully, say so.
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        """
        Initialize LLM with the specified model.
        
        Args:
            model_name: Name of the Llama model to use
        """
        self._llm = ChatOllama(model=model_name, temperature=0.3)
        self._rag_prompt = ChatPromptTemplate.from_template(self.RAG_TEMPLATE)
    
    @staticmethod
    def format_history(h: BaseChatMessageHistory) -> str:
        """
        Helper method which formats the provided message history to be inputted as additional
        context to the RAG pipeline.

        Args:
            h: Abstract base class for storing chat message history.
        """
        lines = []
        for m in h.messages:
            if m.type == "human":
                lines.append(f"User: {m.content}")
            elif m.type == "ai":
                lines.append(f"Assistant: {m.content}")
            else:
                lines.append(f"{m.type.capitalize()}: {m.content}")
        return "\n".join(lines)

    def _build_rag_chain(self, retriever: VectorStoreRetriever, history: BaseChatMessageHistory):
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
        conversation_prompt = ChatPromptTemplate.from_template(self.RAG_TEMPLATE)

        return (
            {
                "history": RunnableLambda(lambda _: self.format_history(history)),
                "context": mq_retriever,
                "question": RunnablePassthrough(),
            }
            | conversation_prompt
            | self._llm
            | StrOutputParser()
        )
    
    async def askQuestion(self, question: str, retriever: VectorStoreRetriever, history: BaseChatMessageHistory) -> str:
        """
        Process a question using the RAG pipeline.
        
        Args:
            question: The question to answer
            retriever: Vector store retriever
            history: Additional context of message history
            
        Returns:
            Generated answer string
        """
        try:
            logger.info("Building RAG chain with conversation history")
            chain = self._build_rag_chain(retriever, history)
            logger.info("Generating response")
            response = await chain.ainvoke(question)
            logger.info("Response generated successfully")
            return response
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise