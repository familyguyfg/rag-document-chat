"""
CONFIG - All settings in one place.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# --- LLM Settings ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# --- Chunking Settings ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Vector Store ---
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "documents"

# --- Retrieval ---
TOP_K = 5

# --- Agent ---
MAX_STEPS = 10