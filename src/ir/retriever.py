import re

import numpy as np
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def build_bm25_index(self):
        results = self.collection.get(include=["documents"])
        self.documents = results["documents"]
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.doc_index = {doc: i for i, doc in enumerate(self.documents)}

    def hybrid_search(self, query: str, alpha: float = 0.5, top_k: int = 10, service_filter=None):
        vector_top_k = 30
        bm25_top_k = 30

        # --- Vector search ---
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=vector_top_k,
            where={"service": service_filter},
        )

        vector_docs = vector_results["documents"][0]
        vector_scores = [1 - d for d in vector_results["distances"][0]]

        # --- BM25 search ---
        tokenized_query = self._tokenize(query)
        bm25_scores_full = self.bm25.get_scores(tokenized_query)

        bm25_top_indices = np.argsort(bm25_scores_full)[::-1][:bm25_top_k]
        bm25_docs = [self.documents[i] for i in bm25_top_indices]

        # --- Union ---
        candidate_docs = list(set(vector_docs + bm25_docs))

        hybrid_results = []

        for doc in candidate_docs:
            idx = self.doc_index[doc]
            vec_score = 0
            if doc in vector_docs:
                vec_score = vector_scores[vector_docs.index(doc)]

            bm_score = bm25_scores_full[idx]
            hybrid_score = alpha * vec_score + (1 - alpha) * bm_score

            hybrid_results.append(
                {
                    "text": doc,
                    "vector_score": vec_score,
                    "bm25_score": bm_score,
                    "hybrid_score": hybrid_score,
                }
            )

        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:top_k]
