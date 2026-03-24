from pathlib import Path

# Project root: src/ir/config.py -> src/ir -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Paths ---
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
ASSETS_PATH = PROJECT_ROOT / "assets"
EVAL_OUTPUT_FILE = PROJECT_ROOT / "eval_queries_ag.jsonl"

# --- Chroma ---
COLLECTION_NAME = "aws_docs"

# --- Models ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLAMA_MODEL = "llama3"
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# --- Hyperparameters ---
PARSER_MAX_TOKENS = 400
QUERY_REWRITER_N_QUERIES = 2
SERVICE_PREDICTOR_TOP_K = 2
RETRIEVER_ALPHA = 0.5
RETRIEVER_VECTOR_TOP_K = 30
RETRIEVER_BM25_TOP_K = 30
RETRIEVER_HYBRID_TOP_K = 20
RERANKER_TOP_K = 5
EVAL_TOP_K = 10
