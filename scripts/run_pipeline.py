import pandas as pd
import chromadb

import ir.config as cfg
from ir.pipeline import RAGPipeline

QUERIES = [
    "how to set up basic execution role for lambda function",
    "s3 bucket versioning enable and configure",
    "dynamodb partition key and sort key design best practices",
    "sns topic publish message example",
    "sqs queue visibility timeout configuration",
    "api gateway lambda proxy integration setup",
    "iam policy attach to role permissions",
    "cloudformation stack create template example",
    "glue crawler create and run on s3 data",
    "step functions state machine define and execute",
]


def confidence_label(top_score: float, second_score: float) -> str:
    margin = top_score - second_score
    if top_score >= 6.5 and margin > 0.5:
        return "HIGH"
    elif top_score >= 5.0:
        return "MEDIUM"
    else:
        return "LOW"


def run(queries: list[str]) -> pd.DataFrame:
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_PATH))
    collection = client.get_collection(name=cfg.COLLECTION_NAME)
    rag = RAGPipeline(collection)

    rows = []
    for query in queries:
        output = rag.query(query)
        service = output["service"]
        results = output["results"]

        if not results:
            continue

        top = results[0]
        top1 = round(top["rerank_score"], 3)
        top2 = round(results[1]["rerank_score"], 3) if len(results) > 1 else None

        rows.append({
            "query": query,
            "predicted_service": service,
            "vector_score": round(top["vector_score"], 3),
            "bm25_score": round(top["bm25_score"], 3),
            "hybrid_score": round(top["hybrid_score"], 3),
            "top1_rerank": top1,
            "top2_rerank": top2,
            "confidence": confidence_label(top1, top2) if top2 is not None else "N/A",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run(QUERIES)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df.to_string(index=False))
