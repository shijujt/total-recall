from pathlib import Path

import chromadb

from ir.evaluator import RetrievalEvaluator
from ir.pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
EVAL_FILE = PROJECT_ROOT / "eval_queries_ag.jsonl"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name="aws_docs")

evaluator = RetrievalEvaluator(RAGPipeline(collection))
metrics = evaluator.evaluate(str(EVAL_FILE), top_k=10)
print(metrics)
