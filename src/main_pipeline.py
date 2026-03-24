from rag_pipeline import RAGPipeline
import chromadb

client = chromadb.PersistentClient(path="../chroma_db")
collection_nm = "aws_docs"
collection = client.get_collection(name=collection_nm)

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
scores = [ round(r["rerank_score"], 3) for r in rslts]
scores.sort(reverse=True)
conf_scr = confidence_label(scores[0], scores[1])
print(f"{q}, top_score: {scores[0]}, second_score: {scores[1]}, margin: {round(scores[0]-scores[1], 3)}, {conf_scr}")

