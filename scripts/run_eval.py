import chromadb
import pandas as pd

import ir.config as cfg
from ir.evaluator import RetrievalEvaluator
from ir.pipeline import RAGPipeline

client = chromadb.PersistentClient(path=str(cfg.CHROMA_PATH))
collection = client.get_collection(name=cfg.COLLECTION_NAME)

output = RetrievalEvaluator(RAGPipeline(collection)).evaluate(
    str(cfg.EVAL_OUTPUT_FILE), top_k=cfg.EVAL_TOP_K
)

df = pd.DataFrame(output["per_query"])
pd.set_option("display.max_colwidth", 80)
print("\n--- Per-Query Report ---")
print(df.to_string(index=False))

print("\n--- Summary ---")
for k, v in output["summary"].items():
    print(f"  {k}: {v}")
