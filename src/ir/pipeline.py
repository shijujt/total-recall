from ir.qr.query_writer import LlamaQueryRewriter
from ir.reranker import Reranker
from ir.retriever import HybridRetriever
from ir.service_predictor import ServicePredictor


class RAGPipeline:
    def __init__(self, chroma_collection):
        self.hybrid = HybridRetriever(chroma_collection)
        self.rewriter = LlamaQueryRewriter(2)
        self.reranker = Reranker()
        self.service_predictor = ServicePredictor()

        print("Building BM25 index...")
        self.hybrid.build_bm25_index()
        print("Ready.")

    def query(self, query: str, top_k: int = 5):
        # STEP 1 — Predict service
        services = self.service_predictor.predict(query, top_k=2)
        print("\nPredicted services:", services)
        best_service = services[0][0]

        # Step 2: Query rewriting
        rewritten_queries = self.rewriter.rewrite(query)

        print("\n------------->>>>> Rewritten queries: <<<<<<----------\n")
        for q in rewritten_queries:
            print(" -", q)

        # Step 3: Retrieve for each query
        all_candidates = []
        for q in rewritten_queries:
            results = self.hybrid.hybrid_search(q, alpha=0.5, top_k=20, service_filter=best_service)
            all_candidates.extend(results)

        # Step 4: Deduplicate
        unique = {}
        for c in all_candidates:
            unique[c["text"]] = c

        candidates = list(unique.values())

        # Step 5: Rerank using ORIGINAL query
        reranked_results = self.reranker.rerank(query, candidates, top_k=top_k)

        print("\n==============================")
        print("Query:", query)
        print("==============================")
        print("\n--- FINAL RESULTS (Hybrid vs Rerank) ---")

        for result in reranked_results:
            print("\nHybrid Score :", round(result["hybrid_score"], 3))
            print("Vector Score :", round(result["vector_score"], 3))
            print("BM25 Score   :", round(result["bm25_score"], 3))
            print("Rerank Score :", round(result["rerank_score"], 3))
            print(result["text"][:100])
            print("-" * 60)

        return reranked_results
