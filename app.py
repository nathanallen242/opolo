import streamlit as st
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory

from app.core.document import DocumentProcessor
from app.core.db import DocumentStore
from app.core.llm import LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="opolo",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .stAppHeader {
            display: none;
        }
        .stApp {
            margin-top: 80px;
        }

        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background-color: #0E1117;
            padding: 1rem;
            z-index: 999;
            border-bottom: 1px solid #333;
        }

        .main-content {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }

        .chat-message {
            max-width: 600px;
            width: 100%;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
        }

        .chat-message.user {
            background-color: #2E303D;
        }

        .chat-message.assistant {
            background-color: #1F2937;
        }

        .chat-avatar {
            width: 2.5rem;
            height: 2.5rem;
            margin-right: 1rem;
            border-radius: 0.5rem;
        }

        .chat-content {
            flex-grow: 1;
        }

        /* Welcome section styling */
        .welcome-container {
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
        }

        .sample-questions {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .question-box {
            background: #2E303D;
            padding: 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            flex: 1;
            text-align: center;
            transition: background-color 0.3s;
        }

        .question-box:hover {
            background: #3E4049;
        }

        /* Form styling */
        [data-testid="stForm"] {
            background: #1E1E1E;
            padding: 15px;
            border-radius: 10px;
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 800px;
        }

        .send-icon {
            color: white;
            cursor: pointer;
            padding: 8px;
            border-radius: 8px;
            background: #1F61D5;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 40px;
        }

        .send-icon:hover {
            background: #1850B5;
        }

        /* Hide Streamlit elements */
        .stDeployButton, footer, #MainMenu, .stDecoration {
            display: none !important;
        }

        .stTextInput > div > div {
            padding-top: 0 !important;
        }

        .stTextInput [data-baseweb="input"] {
            background: #2E303D;
            border-color: #4A4B53;
            border-radius: 8px;
        }

        /* PDF viewer styling */
        .pdf-viewer {
            height: calc(100vh - 100px);
            width: 100%;
            border: none;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

class StreamlitApp:
    """Streamlit interface for document question answering."""
    
    def __init__(self, data_dir: str = "./data", persist_dir: str = "./chroma_db"):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.doc_processor = DocumentProcessor()
        self.doc_store = None
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "first_query" not in st.session_state:
            st.session_state.first_query = True

        if "doc_store_initialized" not in st.session_state:
            st.session_state.doc_store_initialized = False

        try:
            with st.container():
                st.markdown('<div style="margin-top: 80px;">', unsafe_allow_html=True)

                if not st.session_state.doc_store_initialized:
                    persist_path = Path(self.persist_dir)
                    db_exists = persist_path.exists() and any(persist_path.iterdir())
                    
                    if db_exists:
                        st.info("Loading existing document database...")
                        st.session_state.doc_store = DocumentStore(
                            persist_directory=str(self.persist_dir),
                            load_from_disk=True
                        )
                        st.success("Successfully loaded existing document database")
                    else:
                        st.info("Initializing new document database...")
                        persist_path.mkdir(parents=True, exist_ok=True)
                        
                        st.session_state.doc_store = DocumentStore(
                            persist_directory=str(self.persist_dir),
                            load_from_disk=False
                        )
                        
                        self.doc_store = st.session_state.doc_store
                        self._process_documents()
                        st.success("Successfully initialized and processed new document database")
                    
                    st.session_state.doc_store_initialized = True
                
                self.doc_store = st.session_state.doc_store
                
                st.markdown('</div>', unsafe_allow_html=True)
        
            self.llm = LLM() if self.doc_store else None

        except Exception as e:
            st.error(f"Failed to initialize application: {str(e)}")
            logger.error(f"Initialization error: {e}")
            raise

    def _process_documents(self) -> None:
        """Process PDF documents and add them to the vector store."""
        try:
            pdf_files = list(self.data_dir.glob("*.pdf"))
            if not pdf_files:
                st.error(f"No PDF files found in {self.data_dir}")
                return
            
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_chunks = []
                for idx, pdf_file in enumerate(pdf_files):
                    status_text.text(f"Processing: {pdf_file.name}")
        
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
                    progress_bar.progress((idx + 1) / len(pdf_files))
                
                status_text.text("Adding documents to vector store...")
                self.doc_store.add_documents(all_chunks)
                progress_bar.empty()
                status_text.empty()
                st.success("Successfully processed and stored all documents")
                
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            st.error(f"Failed to process documents: {str(e)}")
            raise
    
    def display_chat_message(self, message: str, is_user: bool = False) -> None:
        """Display a chat message in the UI."""
        avatar = "👤" if is_user else "🤖"
        class_name = "user" if is_user else "assistant"
        
        st.markdown(f"""
            <div class="chat-message {class_name}">
                <div class="chat-avatar">{avatar}</div>
                <div class="chat-content">{message}</div>
            </div>
        """, unsafe_allow_html=True)
    
    def display_welcome(self) -> None:
        """Display the welcome message and sample questions."""
        # Apply custom CSS for styling
        st.markdown("""
            <style>
                .welcome-container {
                    text-align: center;
                    padding: 50px;
                }
                .welcome-container h1 {
                    font-size: 3em;
                    margin-bottom: 20px;
                }
                .welcome-container p {
                    font-size: 1.5em;
                    margin-bottom: 40px;
                }
                .sample-questions {
                    display: flex;
                    justify-content: center;
                    gap: 20px;
                }
                .question-button {
                    background-color: #f0f2f6;
                    border: 2px solid #007BFF;
                    border-radius: 10px;
                    padding: 20px;
                    width: 250px;
                    height: 100px;
                    font-size: 1.2em;
                    cursor: pointer;
                    transition: background-color 0.3s, transform 0.3s;
                }
                .question-button:hover {
                    background-color: #e0e5ec;
                    transform: scale(1.05);
                }
                /* Hide the default Streamlit button styles */
                .stButton > button {
                    all: unset;
                }
            </style>
        """, unsafe_allow_html=True)

        # Welcome message
        st.markdown("""
            <div class="welcome-container">
                <h1>👋 Welcome to Opolo!</h1>
                <p>Get started by asking a question about your documents</p>
            </div>
        """, unsafe_allow_html=True)

        # Sample questions
        sample_questions = [
            {
                "label": "What are the main topics covered?",
                "question": "What are the main topics covered in the documents?"
            },
            {
                "label": "Summarize key findings",
                "question": "Can you summarize the key findings?"
            },
            {
                "label": "What are the conclusions?",
                "question": "What are the main conclusions?"
            },
        ]

        # Display sample questions as styled buttons
        cols = st.columns(len(sample_questions))
        for col, sample in zip(cols, sample_questions):
            with col:
                # Use a container to apply custom styles to the button
                if st.button(sample["label"], key=sample["label"]):
                    asyncio.run(self.process_query(sample["question"]))

    async def process_query(self, query: str) -> None:
        """Process a user query and update chat history."""
        try:
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.first_query = False
            
            chat_history = ChatMessageHistory()
            for msg in st.session_state.messages[:-1]: 
                if msg["role"] == "user":
                    chat_history.add_user_message(msg["content"])
                else:
                    chat_history.add_ai_message(msg["content"])

            with st.spinner("Thinking..."):
                answer = await self.llm.askQuestion(
                    query,
                    self.doc_store.get_retriever(),
                    chat_history
                )
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            st.session_state.messages.pop()
            st.error("An error occurred while processing your question. Please try again.")

    def render(self) -> None:
        """Render the main application interface."""
        st.markdown('<div class="header"><h1>🧠 opolo</h1></div>', unsafe_allow_html=True)

        with st.sidebar:
            st.markdown("### 📄 PDF Viewer")
            pdf_files = list(self.data_dir.glob("*.pdf"))
            if pdf_files:
                selected_pdf = st.selectbox(
                    "Select a document to view",
                    options=pdf_files,
                    format_func=lambda x: x.name
                )
                if selected_pdf:
                    st.markdown(f'<iframe class="pdf-viewer" src="data/{selected_pdf.name}"></iframe>', unsafe_allow_html=True)

        st.markdown('<div class="main-content">', unsafe_allow_html=True)

        if st.session_state.first_query:
            self.display_welcome()

        for message in st.session_state.messages:
            self.display_chat_message(
                message["content"],
                is_user=(message["role"] == "user")
            )

        with st.form(key="chat_form", clear_on_submit=True):
            cols = st.columns([5, 1])
            with cols[0]:
                query = st.text_input(
                    "",
                    key="chat_input",
                    placeholder="Type your question here..."
                )
            with cols[1]:
                submit = st.form_submit_button(
                    "📤",
                    help="Send message",
                    use_container_width=True
                )
            
            if submit and query:
                asyncio.run(self.process_query(query))
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

def main() -> None:
    """Main entry point for the application."""
    load_dotenv()
    app = StreamlitApp()
    app.render()

if __name__ == "__main__":
    main()