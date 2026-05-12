"""
Configuration Settings for Nalpamaradi Keram Content Generation System
Complete system configuration with validation and persistence settings
"""

import os
from pathlib import Path

# ============================================================================
# PRODUCT INFO
# ============================================================================

PRODUCT_NAME = "Nalpamaradi Keram"
BRAND_NAME = "Kerala Ayurveda"
PRODUCT_CATEGORY = "Ayurvedic Skincare Oil"

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Groq API Key (REQUIRED)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Model (Groq)
LLM_MODEL = "llama-3.3-70b-versatile"

# ============================================================================
# PDF PROCESSING CONFIGURATION
# ============================================================================

# OCR Settings
OCR_ENABLED = False
OCR_MIN_TEXT_LENGTH = 30

# Chunking Settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 100

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

TOP_K_RETRIEVAL = 5
BM25_CANDIDATES = 30
SIMILARITY_THRESHOLD = 0.3

# Retrieval boost settings - DISABLED for balanced retrieval
PRODUCT_INFO_BOOST_AUTO = 1.0  # No boost
PRODUCT_INFO_BOOST_FORCED = 1.0  # No boost

# Hybrid search weights
BM25_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.7

# Intent-aware retrieval settings
PRODUCT_FIRST_RETRIEVAL = 2  # Get 2 product chunks first
PDF_ENRICHMENT_RETRIEVAL = 3  # Then get 3 PDF chunks for context

# ============================================================================
# GENERATION SETTINGS
# ============================================================================

# Temperatures
OUTLINE_TEMPERATURE = 0.4
WRITER_TEMPERATURE = 0.5
FACT_CHECKER_TEMPERATURE = 0.2

# Retry logic
MAX_RETRIES = 3

# ============================================================================
# SAFETY PATTERNS
# ============================================================================

MEDICAL_CLAIM_PATTERNS = [
    r"\b(cure|cures|cured|curing)\b",
    r"\b(treat|treats|treated|treating|treatment of)\b",
    r"\b(prevent|prevents|prevented|preventing|prevention of)\b",
    r"\b(diagnose|diagnosis)\b",
    r"\b(reverse|reverses|reversed)\b",
    r"\b(eliminate|eliminates|eliminated)\b",
]

PROHIBITED_MEDICAL_CLAIMS = MEDICAL_CLAIM_PATTERNS + [
    r"\b(guaranteed|guarantee|scientifically proven)\b",
]

SAFE_LANGUAGE = [
    "may support",
    "traditionally used for",
    "can help with",
    "associated with",
    "known to nourish",
    "may contribute to",
]

# ============================================================================
# STORAGE & CACHING - ENHANCED FOR PERSISTENCE
# ============================================================================

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"
CHROMA_DIR = DATA_DIR / "chroma_db"  # Persistent vector DB

for directory in [DATA_DIR, PDF_DIR, CACHE_DIR, OUTPUT_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ChromaDB - Persistent Settings
CHROMA_COLLECTION_NAME = "nalpamaradi_corpus"
CHROMA_DISTANCE_METRIC = "cosine"
CHROMA_PERSIST_DIRECTORY = str(CHROMA_DIR)

# Corpus Management
ENABLE_CACHE = True
CACHE_EMBEDDINGS = True
CACHE_LLM_RESPONSES = False
PERSIST_CORPUS = True  # NEW: Enable persistent corpus
CORPUS_METADATA_FILE = CACHE_DIR / "corpus_metadata.json"  # Track processed files

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE = DATA_DIR / "system.log"

SHOW_PROGRESS_BARS = True
LOG_CHUNK_STATISTICS = True

# ============================================================================
# PERFORMANCE
# ============================================================================

EMBEDDING_BATCH_SIZE = 32
LLM_REQUEST_TIMEOUT = 60
MAX_REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 0.1
MAX_CHUNKS_IN_MEMORY = 10000
CLEAR_CACHE_ON_RESTART = False

# ============================================================================
# VALIDATION
# ============================================================================


def validate_config():
    """Validate configuration settings"""

    if GROQ_API_KEY == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not configured. Please set it as an environment variable or in config.py"
        )

    if OCR_ENABLED:
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "OCR is enabled but dependencies are missing. "
                "Install with: pip install pytesseract pdf2image"
            )

    if CHUNK_SIZE <= CHUNK_OVERLAP:
        raise ValueError(
            f"CHUNK_SIZE ({CHUNK_SIZE}) must be greater than CHUNK_OVERLAP ({CHUNK_OVERLAP})"
        )

    if TOP_K_RETRIEVAL > BM25_CANDIDATES:
        raise ValueError(
            f"TOP_K_RETRIEVAL ({TOP_K_RETRIEVAL}) cannot exceed BM25_CANDIDATES ({BM25_CANDIDATES})"
        )

    if abs((BM25_WEIGHT + SEMANTIC_WEIGHT) - 1.0) > 0.01:
        raise ValueError("BM25_WEIGHT + SEMANTIC_WEIGHT must equal 1.0")

    print("✓ Configuration validated successfully")


# ============================================================================
# SYSTEM METADATA
# ============================================================================

SYSTEM_NAME = "Nalpamaradi Keram Agentic Content System"
SYSTEM_VERSION = "2.1.0"
RELEASE_DATE = "2024-12-17"

FEATURES = [
    "Persistent vector database",
    "Intent-aware Q&A with corpus type awareness",
    "Page-level PDF isolation",
    "OCR fallback",
    "Hybrid BM25 + semantic retrieval",
    "Agentic workflow with retries",
    "Medical claim safety enforcement",
]

# ============================================================================
# SUMMARY
# ============================================================================


def get_config_summary():
    return {
        "system": {
            "name": SYSTEM_NAME,
            "version": SYSTEM_VERSION,
            "brand": BRAND_NAME,
            "product": PRODUCT_NAME,
        },
        "models": {
            "embedding": EMBEDDING_MODEL,
            "llm": LLM_MODEL,
        },
        "retrieval": {
            "top_k": TOP_K_RETRIEVAL,
            "bm25_candidates": BM25_CANDIDATES,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
        "persistence": {
            "enabled": PERSIST_CORPUS,
            "chroma_dir": CHROMA_PERSIST_DIRECTORY,
        },
    }


# ============================================================================
# RUNTIME CHECK
# ============================================================================

if __name__ == "__main__":
    validate_config()
