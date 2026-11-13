from pathlib import Path
import os

# Project Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FRAUD_RULES_PATH = DATA_DIR / "fraud_rules.json"
BANK_TRANSACTIONS_PATH = DATA_DIR / "bank_transactions_data_2.csv"

# Model Configurations (with defaults overridable via environment variables)
EMBEDDING_MODEL_NAME = os.getenv("embedding_model_name", os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text"))
LLM_MODEL_NAME = os.getenv("llm_model_name", os.getenv("LLM_MODEL_NAME", "llama3.1"))
LLM_TEMPERATURE = float(os.getenv("llm_model_temperature", os.getenv("LLM_TEMPERATURE", "0.5")))

# Cosine similarity threshold for RAG semantic rule matching (0.0 to 1.0)
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.45"))
RAG_TOP_K_RULES = int(os.getenv("RAG_TOP_K_RULES", "3"))

# Resilience Settings
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
OLLAMA_BASE_DELAY = float(os.getenv("OLLAMA_BASE_DELAY", "0.5"))
