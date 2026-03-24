import ir.config as cfg
from ir.qr.query_writer import LlamaQueryRewriter
from ir.reranker import Reranker
from ir.retriever import HybridRetriever
from ir.service_predictor import ServicePredictor


class RAGPipeline:
    def __init__(self, chroma_collection):
        self.hybrid = HybridRetriever(chroma_collection)
        self.rewriter = LlamaQueryRewriter(cfg.QUERY_REWRITER_N_QUERIES)
        self.reranker = Reranker()
        self.service_predictor = ServicePredictor()

        print("Building BM25 index...")
        self.hybrid.build_bm25_index()
        print("Ready.")

    def query(self, query: str, top_k: int = cfg.RERANKER_TOP_K) -> dict:
        services = self.service_predictor.predict(query, top_k=cfg.SERVICE_PREDICTOR_TOP_K)
        best_service = services[0][0]

        rewritten_queries = self.rewriter.rewrite(query)

        all_candidates = []
        for q in rewritten_queries:
            results = self.hybrid.hybrid_search(
                q,
                alpha=cfg.RETRIEVER_ALPHA,
                top_k=cfg.RETRIEVER_HYBRID_TOP_K,
                service_filter=best_service,
            )
            all_candidates.extend(results)

        unique = {}
        for c in all_candidates:
            unique[c["text"]] = c

        candidates = list(unique.values())
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        return {"service": best_service, "results": reranked}
