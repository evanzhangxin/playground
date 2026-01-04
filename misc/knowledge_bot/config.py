import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "data")
# Model configuration from .env
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/mnt/e/Models/all-MiniLM-L6-v2")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "./models")
