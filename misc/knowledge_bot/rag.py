import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import config

class KnowledgeBase:
    def __init__(self, persist_directory=config.CHROMA_DB_DIR):
        self.persist_directory = persist_directory
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = None
        self.qa_chain = None
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.persist_directory):
            self.vector_db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embedding_function
            )
            self._init_qa_chain()
        else:
            # Initialize empty db
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )

    def _init_qa_chain(self):
        # Prefer DashScope (Qwen) if available, otherwise fallback to OpenAI
        if self.vector_db:
            llm = None
            if config.DASHSCOPE_API_KEY:
                # Use Qwen via DashScope (compatible with OpenAI format if using LangChain's ChatOpenAI with base_url)
                # Or use native implementation. Here we use ChatOpenAI with Qwen compatible endpoint or DashScope direct integration
                # Simpler: Use ChatOpenAI pointing to DashScope/DeepSeek endpoint if possible, 
                # but DashScope has its own library. Let's use ChatOpenAI with DeepSeek/Moonshot base_url pattern for broad compatibility if user sets it.
                # For now, let's assume we use DashScope's Qwen-Turbo via LangChain Community or just standard OpenAI client with changed base_url
                
                # Using Qwen via OpenAI compatible interface (DashScope supports it now)
                llm = ChatOpenAI(
                    api_key=config.DASHSCOPE_API_KEY,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    model="qwen-turbo",
                    temperature=0
                )
            elif config.OPENAI_API_KEY:
                llm = ChatOpenAI(temperature=0, api_key=config.OPENAI_API_KEY)
            
            if llm:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )

    def add_document(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        if self.vector_db:
            self.vector_db.add_documents(texts)
        else:
            self.vector_db = Chroma.from_documents(
                documents=texts, 
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )
        
        # Re-init QA chain to ensure it uses the updated DB if needed (usually retriever is live)
        if not self.qa_chain:
            self._init_qa_chain()
            
        return len(texts)

    def query(self, question):
        if not self.qa_chain:
            return "Knowledge base not initialized or API Key (OpenAI/DashScope) missing."
        
        try:
            result = self.qa_chain.invoke({"query": question})
            return result["result"]
        except Exception as e:
            return f"Error querying knowledge base: {str(e)}"

    def clear(self):
        if self.vector_db:
            self.vector_db.delete_collection()
            self.vector_db = None
            # Re-create empty
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )

if __name__ == "__main__":
    kb = KnowledgeBase()
    # kb.add_document("data/example.txt")
    # print(kb.query("What is in the text?"))
