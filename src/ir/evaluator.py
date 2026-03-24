import json

import ir.config as cfg


class RetrievalEvaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def load_queries(self, filepath):
        queries = []
        with open(filepath, "r") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON error on line {i}")
                    print(line)
                    raise e

        return queries

    def is_relevant(self, chunk_text, keywords):
        text = chunk_text.lower()

        for kw in keywords:
            if kw.lower() in text:
                return True

        return False

    def evaluate(self, eval_file, top_k=cfg.EVAL_TOP_K):
        eval_queries = self.load_queries(eval_file)

        hits = 0
        recall_hits = 0
        reciprocal_sum = 0

        total = len(eval_queries)
        for item in eval_queries:
            query = item["query"]
            keywords = item["keywords"]
            results = self.pipeline.query(query, top_k=top_k)["results"]
            rank = None
            for i, r in enumerate(results):
                if self.is_relevant(r["text"], keywords):
                    rank = i + 1
                    break

            if rank:
                recall_hits += 1
                if rank == 1:
                    hits += 1

                reciprocal_sum += 1 / rank

        recall = recall_hits / total
        hit_rate = hits / total
        mrr = reciprocal_sum / total

        return {
            "total_queries": total,
            f"Recall@{top_k}": round(recall, 3),
            "HitRate@1": round(hit_rate, 3),
            "MRR": round(mrr, 3),
        }
