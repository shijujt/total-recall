from rag_ret_eval import RetrievalEvaluator
from rag_pipeline import RAGPipeline
import chromadb

client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection(name="aws_docs")

evaluator = RetrievalEvaluator(RAGPipeline(collection))
metrics = evaluator.evaluate("../eval_queries_ag.jsonl", top_k=10)
print(metrics)
