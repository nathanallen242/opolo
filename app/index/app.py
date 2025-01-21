import asyncio
import logging
import sys

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from app.core.document import DocumentProcessor
from app.core.db import DocumentStore
from app.core.llm import LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentQA:
    """CLI application for document question answering."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize components."""
        self.data_dir = Path(data_dir)
        self.doc_processor = DocumentProcessor()
        self.doc_store = DocumentStore()
        self.llm = LLM()
        
    async def initialize(self):
        """Load and process documents into vector store."""
        try:
            pdf_files = list(self.data_dir.glob("*.pdf"))
            if not pdf_files:
                logger.error(f"No PDF files found in {self.data_dir}")
                sys.exit(1)
                
            logger.info("Loading and processing documents...")
            
            all_chunks = []
            for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
                docs = self.doc_processor.load_pdf(pdf_file)
                chunks = self.doc_processor.split_documents(docs)
                
                processed_chunks = [
                    {
                        'content': chunk.page_content,
                        'metadata': {
                            **chunk.metadata,
                            'source': pdf_file.name
                        }
                    }
                    for chunk in chunks
                ]
                all_chunks.extend(processed_chunks)
            
            logger.info("Adding documents to vector store...")
            self.doc_store.add_documents(all_chunks)
            
            logger.info(f"Initialized with {len(all_chunks)} document chunks")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            sys.exit(1)

    async def run_cli(self):
        """Run the CLI interface."""
        print("\nDocument QA System")
        print("Enter 'q' to quit\n")
        
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                
                if question.lower() == 'q':
                    print("\nExiting...")
                    break
                    
                if not question:
                    continue
                
                with tqdm(total=1, desc="Generating answer") as pbar:
                    answer = await self.llm.askQuestion(
                        question,
                        self.doc_store.get_retriever()
                    )
                    pbar.update(1)
                
                print(f"\nAnswer: {answer}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print("\nAn error occurred. Please try again.")

async def main():
    """Main entry point."""
    load_dotenv()

    app = DocumentQA()
    print("Initializing system...")
    await app.initialize()
    await app.run_cli()

if __name__ == "__main__":
    asyncio.run(main())