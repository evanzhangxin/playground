import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config

class KnowledgeBase:
    def __init__(self, persist_directory=config.CHROMA_DB_DIR):
        logger.info("Initializing KnowledgeBase...")
        self.persist_directory = persist_directory
        logger.info("Loading embedding model...")
        try:
            logger.info("Creating HuggingFaceEmbeddings instance...")
            
            # Use model paths from config
            model_paths = [
                config.EMBEDDING_MODEL_PATH,  # User's preferred path from config
                config.EMBEDDING_CACHE_DIR  # Cache folder from config
            ]
            
            # First check if any local path exists
            local_model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    local_model_path = path
                    logger.info("Found local model at: %s", local_model_path)
                    break
            
            if local_model_path:
                # Use local model path
                self.embedding_function = HuggingFaceEmbeddings(
                    model_name=local_model_path,
                    model_kwargs={"device": "cpu"}  # Force CPU for faster loading
                )
            else:
                # Fallback to model name from config
                logger.info("No local model found, using model name: %s", config.EMBEDDING_MODEL_NAME)
                self.embedding_function = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL_NAME,
                    cache_folder=config.EMBEDDING_CACHE_DIR,
                    model_kwargs={"device": "cpu"}  # Force CPU for faster loading
                )
            
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error("Error loading embedding model: %s", str(e), exc_info=True)
            raise
        self.vector_db = None
        self.qa_chain = None
        self._load_db()
        logger.info("KnowledgeBase initialized successfully")

    def _load_db(self):
        logger.info("Loading database from %s...", self.persist_directory)
        if os.path.exists(self.persist_directory):
            logger.info("Found existing database, loading...")
            self.vector_db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embedding_function
            )
            logger.info("Database loaded successfully")
            self._init_qa_chain()
        else:
            # Initialize empty db
            logger.info("No existing database found, initializing empty database...")
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            logger.info("Empty database initialized")

    def _init_qa_chain(self):
        logger.info("Initializing QA chain...")
        # Prefer DashScope (Qwen) if available, otherwise fallback to OpenAI
        if self.vector_db:
            llm = None
            if config.DASHSCOPE_API_KEY:
                logger.info("Using DashScope (Qwen) LLM")
                # Using Qwen via OpenAI compatible interface (DashScope supports it now)
                llm = ChatOpenAI(
                    api_key=config.DASHSCOPE_API_KEY,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    model="qwen-turbo",
                    temperature=0
                )
                logger.info("DashScope LLM initialized")
            elif config.OPENAI_API_KEY:
                logger.info("Using OpenAI LLM")
                llm = ChatOpenAI(temperature=0, api_key=config.OPENAI_API_KEY)
                logger.info("OpenAI LLM initialized")
            else:
                logger.warning("No API keys found, cannot initialize QA chain")
                return
            
            if llm:
                logger.info("Creating QA chain with LCEL...")
                prompt = ChatPromptTemplate.from_template(
                    "Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"
                )
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                self.qa_chain = (
                    {
                        "context": self.vector_db.as_retriever(search_kwargs={"k": 3}) | format_docs,
                        "question": RunnablePassthrough()
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                logger.info("QA chain initialized successfully")
        else:
            logger.warning("Vector DB not available, cannot initialize QA chain")

    def add_document(self, file_path):
        logger.info("Processing document: %s", file_path)
        loader = TextLoader(file_path, encoding='utf-8')
        logger.info("Loading document...")
        documents = loader.load()
        logger.info("Document loaded, found %d document(s)", len(documents))
        
        logger.info("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logger.info("Document split into %d chunks", len(texts))
        
        if self.vector_db:
            logger.info("Adding chunks to existing database...")
            self.vector_db.add_documents(texts)
            logger.info("Chunks added successfully")
        else:
            logger.info("Creating new database and adding chunks...")
            self.vector_db = Chroma.from_documents(
                documents=texts, 
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )
            logger.info("New database created with chunks added")
        
        # Re-init QA chain to ensure it uses the updated DB if needed (usually retriever is live)
        if not self.qa_chain:
            logger.info("QA chain not initialized, initializing now...")
            self._init_qa_chain()
            
        logger.info("Document processing completed successfully")
        return len(texts)

    def query(self, question):
        logger.info("Received query: %s", question)
        if not self.qa_chain:
            logger.warning("QA chain not initialized, cannot process query")
            return "Knowledge base not initialized or API Key (OpenAI/DashScope) missing."
        
        try:
            logger.info("Processing query...")
            result = self.qa_chain.invoke(question)
            logger.info("Query processed successfully")
            logger.debug("Query result: %s", result)
            return result
        except Exception as e:
            logger.error("Error processing query: %s", str(e), exc_info=True)
            return f"Error querying knowledge base: {str(e)}"

    def clear(self):
        logger.info("Clearing knowledge base...")
        if self.vector_db:
            logger.info("Deleting existing collection...")
            self.vector_db.delete_collection()
            self.vector_db = None
            # Re-create empty
            logger.info("Re-creating empty database...")
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            logger.info("Knowledge base cleared successfully")
        else:
            logger.warning("No vector database found to clear")

if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.add_document("data/example.txt")
    print(kb.query("What is in the text?"))
