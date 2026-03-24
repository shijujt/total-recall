from pathlib import Path

import chromadb

from ir.pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name="aws_docs")

rag = RAGPipeline(collection)


def confidence_label(top_score, second_score):
    margin = top_score - second_score

    if top_score >= 6.5 and margin > 0.5:
        return "HIGH"
    elif top_score >= 5.0:
        return "MEDIUM"
    else:
        return "LOW"


q = "how to set up basic execution role for lambda function"

rslts = rag.query(q)
scores = [round(r["rerank_score"], 3) for r in rslts]
scores.sort(reverse=True)
conf_scr = confidence_label(scores[0], scores[1])
print(
    f"{q}, top_score: {scores[0]}, second_score: {scores[1]}, "
    f"margin: {round(scores[0] - scores[1], 3)}, {conf_scr}"
)
