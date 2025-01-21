"""LLM operations for question answering and retrieval."""
import logging
from typing import List
from langchain_ollama.chat_models import ChatOllama
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.schema.vectorstore import VectorStore
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableSequence

logger = logging.getLogger(__name__)

class LLM:
    """Handles LLM operations with MultiQueryRetrieval capability."""
    
    def __init__(self, model_name: str = "llama2"):
        """
        Initialize LLM with the specified model.
        
        Args:
            model_name: Name of the Llama model to use.
        """
        self._llm = ChatOllama(model_name=model_name)
        self._query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model tasked with generating multiple search queries 
            for finding relevant information to answer a question. Generate multiple different ways to ask 
            the following question that will help get relevant information from a vector database.
            Make the queries diverse to capture different aspects of the question.

            Original question: {question}

            Generate different versions of the question, separated by commas:
            """
        )
        self._output_parser = CommaSeparatedListOutputParser()
        self._query_pipeline = self._query_prompt | self._llm
        self._answer_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context to answer the question.
            If you cannot answer based on the context, say "I don't have enough information to answer that."

            Context: {context}

            Question: {question}

            Answer:"""
        )

    async def _generate_queries(self, question: str) -> List[str]:
        """
        Generate multiple versions of the input question.
        
        Args:
            question: Original question
        
        Returns:
            List of generated questions.
        """
        raw_text = await self._query_pipeline.ainvoke({"question": question})
        return self._output_parser.parse(raw_text)

    async def askQuestion(self, question: str, retriever: VectorStore) -> str:
        """
        Ask a question using MultiQueryRetriever and produce a final answer.
        
        Args:
            question: The question to answer
            retriever: Vector store to use for document lookup
        
        Returns:
            The final answer string.
        """
        try:
            mq_retriever = MultiQueryRetriever(
                llm=self._llm,
                retriever=retriever.as_retriever(),
                prompt=self._query_prompt
            )
            retrieval_chain = RetrievalQA.from_chain_type(
                llm=self._llm,
                chain_type="stuff",
                retriever=mq_retriever,
                chain_type_kwargs={
                    "prompt": self._answer_prompt
                }
            )

            response = await retrieval_chain.ainvoke({"query": question})
            return response["result"]
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise
