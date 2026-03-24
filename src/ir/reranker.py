from typing import Dict, List

from sentence_transformers import CrossEncoder

import ir.config as cfg


class Reranker:
    def __init__(self, model_name=cfg.RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = cfg.RERANKER_TOP_K):
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]
