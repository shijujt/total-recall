import chromadb

import ir.config as cfg
from ir.evaluator import RetrievalEvaluator
from ir.pipeline import RAGPipeline

client = chromadb.PersistentClient(path=str(cfg.CHROMA_PATH))
collection = client.get_collection(name=cfg.COLLECTION_NAME)

evaluator = RetrievalEvaluator(RAGPipeline(collection))
metrics = evaluator.evaluate(str(cfg.EVAL_OUTPUT_FILE), top_k=cfg.EVAL_TOP_K)
print(metrics)
