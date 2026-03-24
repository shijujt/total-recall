import json
from unittest.mock import MagicMock

import pytest

from ir.evaluator import RetrievalEvaluator


def make_result(text):
    return {"text": text, "rerank_score": 1.0}


def make_pipeline(results_per_call):
    """Return a mock pipeline whose query() returns results_per_call[i] on the i-th call."""
    pipeline = MagicMock()
    pipeline.query.side_effect = results_per_call
    return pipeline


# ---------------------------------------------------------------------------
# is_relevant
# ---------------------------------------------------------------------------


class TestIsRelevant:
    def setup_method(self):
        self.ev = RetrievalEvaluator(MagicMock())

    def test_keyword_found_case_insensitive(self):
        assert self.ev.is_relevant("This text mentions Lambda functions", ["Lambda"])

    def test_keyword_not_found(self):
        assert not self.ev.is_relevant("only s3 content here", ["dynamodb"])

    def test_multiple_keywords_first_matches(self):
        assert self.ev.is_relevant("dynamodb table index", ["dynamo", "s3"])

    def test_multiple_keywords_second_matches(self):
        assert self.ev.is_relevant("configure a lambda handler", ["s3", "lambda"])

    def test_empty_keywords(self):
        assert not self.ev.is_relevant("some text", [])

    def test_empty_text(self):
        assert not self.ev.is_relevant("", ["lambda"])


# ---------------------------------------------------------------------------
# load_queries
# ---------------------------------------------------------------------------


class TestLoadQueries:
    def setup_method(self):
        self.ev = RetrievalEvaluator(MagicMock())

    def test_valid_jsonl(self, tmp_path):
        f = tmp_path / "queries.jsonl"
        records = [
            {"query": "q1", "service": "lambda", "src_doc": "a.md", "keywords": ["kw1"]},
            {"query": "q2", "service": "s3", "src_doc": "b.md", "keywords": ["kw2"]},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records))
        result = self.ev.load_queries(str(f))
        assert len(result) == 2
        assert result[0]["query"] == "q1"

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "queries.jsonl"
        f.write_text(
            json.dumps({"query": "q1", "service": "s3", "src_doc": "a.md", "keywords": []})
            + "\n\n"
            + json.dumps({"query": "q2", "service": "s3", "src_doc": "b.md", "keywords": []})
        )
        result = self.ev.load_queries(str(f))
        assert len(result) == 2

    def test_raises_on_bad_json(self, tmp_path):
        f = tmp_path / "queries.jsonl"
        f.write_text('{"query": "ok"}\nnot valid json\n')
        with pytest.raises(json.JSONDecodeError):
            self.ev.load_queries(str(f))


# ---------------------------------------------------------------------------
# evaluate — metric computation
# ---------------------------------------------------------------------------


def write_eval_file(tmp_path, records):
    f = tmp_path / "eval.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in records))
    return str(f)


def make_query_record(query="q", keywords=None):
    return {"query": query, "service": "lambda", "src_doc": "x.md", "keywords": keywords or ["kw"]}


class TestEvaluateMetrics:
    def test_perfect_recall_and_hit_rate(self, tmp_path):
        records = [make_query_record(keywords=["hit"])] * 2
        path = write_eval_file(tmp_path, records)
        pipeline = make_pipeline(
            [
                [make_result("this text has hit in it")],
                [make_result("another hit here")],
            ]
        )
        ev = RetrievalEvaluator(pipeline)
        metrics = ev.evaluate(path, top_k=5)
        assert metrics["Recall@5"] == 1.0
        assert metrics["HitRate@1"] == 1.0
        assert metrics["MRR"] == 1.0

    def test_zero_recall(self, tmp_path):
        records = [make_query_record(keywords=["missing"])] * 2
        path = write_eval_file(tmp_path, records)
        pipeline = make_pipeline(
            [
                [make_result("no match here")],
                [make_result("also no match")],
            ]
        )
        ev = RetrievalEvaluator(pipeline)
        metrics = ev.evaluate(path, top_k=5)
        assert metrics["Recall@5"] == 0.0
        assert metrics["HitRate@1"] == 0.0
        assert metrics["MRR"] == 0.0

    def test_mrr_hit_at_rank_2(self, tmp_path):
        records = [make_query_record(keywords=["found"])]
        path = write_eval_file(tmp_path, records)
        pipeline = make_pipeline(
            [
                [make_result("no match"), make_result("yes found here")],
            ]
        )
        ev = RetrievalEvaluator(pipeline)
        metrics = ev.evaluate(path, top_k=5)
        assert metrics["MRR"] == 0.5
        assert metrics["HitRate@1"] == 0.0
        assert metrics["Recall@5"] == 1.0

    def test_mrr_hit_at_rank_5(self, tmp_path):
        records = [make_query_record(keywords=["found"])]
        path = write_eval_file(tmp_path, records)
        no_match = [make_result("no match")] * 4
        pipeline = make_pipeline(
            [no_match + [make_result("yes found here")]]
        )
        ev = RetrievalEvaluator(pipeline)
        metrics = ev.evaluate(path, top_k=5)
        assert metrics["MRR"] == round(1 / 5, 3)

    def test_mixed_queries(self, tmp_path):
        records = [
            make_query_record(keywords=["hit"]),
            make_query_record(keywords=["missing"]),
        ]
        path = write_eval_file(tmp_path, records)
        pipeline = make_pipeline(
            [
                [make_result("this has hit in it")],
                [make_result("no match here")],
            ]
        )
        ev = RetrievalEvaluator(pipeline)
        metrics = ev.evaluate(path, top_k=5)
        assert metrics["Recall@5"] == 0.5
        assert metrics["HitRate@1"] == 0.5
        assert metrics["MRR"] == 0.5

    def test_total_queries_key_present(self, tmp_path):
        records = [make_query_record(keywords=["x"])] * 3
        path = write_eval_file(tmp_path, records)
        pipeline = make_pipeline([[make_result("no")] for _ in range(3)])
        ev = RetrievalEvaluator(pipeline)
        metrics = ev.evaluate(path, top_k=5)
        assert metrics["total_queries"] == 3

    def test_metrics_rounded_to_three_decimals(self, tmp_path):
        # 1 hit out of 3 queries at rank 1: MRR = 1/3 ≈ 0.333
        records = [make_query_record(keywords=["hit"])] * 3
        path = write_eval_file(tmp_path, records)
        pipeline = make_pipeline(
            [
                [make_result("has hit")],
                [make_result("no match")],
                [make_result("no match")],
            ]
        )
        ev = RetrievalEvaluator(pipeline)
        metrics = ev.evaluate(path, top_k=5)
        assert metrics["MRR"] == round(1 / 3, 3)
