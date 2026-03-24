import json

import chromadb
import pandas as pd

import ir.config as cfg
from ir.pipeline import RAGPipeline

# Number of queries to sample from the eval file (None = all)
N_QUERIES = 50


def load_eval_queries(path: str, n: int | None = None) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries[:n] if n else queries


def confidence_label(top_score: float, second_score: float) -> str:
    margin = top_score - second_score
    if top_score >= 6.5 and margin > 0.5:
        return "HIGH"
    elif top_score >= 5.0:
        return "MEDIUM"
    else:
        return "LOW"


def is_relevant(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run(eval_queries: list[dict], rag: RAGPipeline) -> pd.DataFrame:
    rows = []
    for item in eval_queries:
        query = item["query"]
        keywords = item.get("keywords", [])

        output = rag.query(query)
        service = output["service"]
        results = output["results"]

        if not results:
            continue

        top = results[0]
        top1 = round(top["rerank_score"], 3)
        top2 = round(results[1]["rerank_score"], 3) if len(results) > 1 else None
        confidence = confidence_label(top1, top2) if top2 is not None else "N/A"

        rows.append({
            "query": query,
            "predicted_service": service,
            "vector_score": round(top["vector_score"], 3),
            "bm25_score": round(top["bm25_score"], 3),
            "hybrid_score": round(top["hybrid_score"], 3),
            "top1_rerank": top1,
            "top2_rerank": top2,
            "confidence": confidence,
            "relevant": is_relevant(top["text"], keywords) if keywords else None,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_PATH))
    collection = client.get_collection(name=cfg.COLLECTION_NAME)
    rag = RAGPipeline(collection)

    eval_queries = load_eval_queries(str(cfg.EVAL_OUTPUT_FILE), n=N_QUERIES)
    df = run(eval_queries, rag)

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("\n--- Per-Query Results ---")
    print(df.to_string(index=False))

    print("\n--- Confidence Calibration (precision of top-1 result by band) ---")
    cal = (
        df.dropna(subset=["relevant"])
        .groupby("confidence")["relevant"]
        .agg(count="count", precision="mean")
        .round(3)
    )
    print(cal.to_string())
