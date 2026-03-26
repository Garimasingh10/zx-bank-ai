import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    
    # Model configuration (DeepSeek-V3 or GPT-4o-mini via OpenRouter generally great for this)
    LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    
    # Paths
    DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
    ESCALATIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "escalations.json")
    
    # Retrieval Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RETRIEVED_DOCS = 3
    DISTANCE_THRESHOLD = 1.2 # Lower means closer match required to avoid hallucination

config = Config()
