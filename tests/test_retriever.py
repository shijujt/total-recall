from unittest.mock import MagicMock

from ir.retriever import HybridRetriever


def make_collection(docs, service="lambda"):
    """Return a mock Chroma collection pre-loaded with the given documents."""
    collection = MagicMock()
    collection.get.return_value = {
        "documents": docs,
        "metadatas": [{"service": service} for _ in docs],
    }
    return collection


def build_retriever(docs):
    """Build a HybridRetriever with BM25 index populated from docs."""
    collection = make_collection(docs)
    retriever = HybridRetriever(collection)
    retriever.build_bm25_index()
    return retriever, collection


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def setup_method(self):
        self.retriever = HybridRetriever(MagicMock())

    def test_lowercases(self):
        assert self.retriever._tokenize("AWS Lambda") == ["aws", "lambda"]

    def test_strips_punctuation(self):
        tokens = self.retriever._tokenize("foo.bar,baz")
        assert tokens == ["foo", "bar", "baz"]

    def test_empty_string(self):
        assert self.retriever._tokenize("") == []

    def test_alphanumeric_preserved(self):
        tokens = self.retriever._tokenize("s3 bucket v2")
        assert tokens == ["s3", "bucket", "v2"]


# ---------------------------------------------------------------------------
# build_bm25_index
# ---------------------------------------------------------------------------


class TestBuildBm25Index:
    def test_bm25_set_after_build(self):
        retriever, _ = build_retriever(["doc one", "doc two", "doc three"])
        assert retriever.bm25 is not None

    def test_documents_populated(self):
        docs = ["doc one", "doc two", "doc three"]
        retriever, _ = build_retriever(docs)
        assert retriever.documents == docs

    def test_doc_index_maps_text_to_position(self):
        docs = ["alpha text", "beta text", "gamma text"]
        retriever, _ = build_retriever(docs)
        for i, doc in enumerate(docs):
            assert retriever.doc_index[doc] == i


# ---------------------------------------------------------------------------
# hybrid_search — score formula and sorting
# ---------------------------------------------------------------------------


def setup_hybrid_search(docs, vector_docs, vector_distances, query="test query"):
    """
    Build a retriever from docs, mock collection.query to return the given
    vector results, and return (retriever, collection).
    """
    retriever, collection = build_retriever(docs)

    collection.query.return_value = {
        "documents": [vector_docs],
        "distances": [vector_distances],
    }
    return retriever, collection


class TestHybridScoreFormula:
    def test_alpha_1_means_pure_vector(self):
        docs = ["lambda execution role setup", "s3 bucket policy configuration"]
        vec_docs = [docs[0]]
        vec_distances = [0.2]  # vector_score = 1 - 0.2 = 0.8

        retriever, _ = setup_hybrid_search(docs, vec_docs, vec_distances)
        results = retriever.hybrid_search("lambda role", alpha=1.0, top_k=10)

        # The vector doc should have hybrid_score == vector_score
        hit = next(r for r in results if r["text"] == docs[0])
        assert abs(hit["hybrid_score"] - hit["vector_score"]) < 1e-6

    def test_alpha_0_means_pure_bm25(self):
        docs = ["lambda execution role setup", "s3 bucket policy configuration"]
        vec_docs = [docs[0]]
        vec_distances = [0.2]

        retriever, _ = setup_hybrid_search(docs, vec_docs, vec_distances)
        results = retriever.hybrid_search("lambda role", alpha=0.0, top_k=10)

        for r in results:
            assert abs(r["hybrid_score"] - r["bm25_score"]) < 1e-6

    def test_alpha_0_5_blends_scores(self):
        docs = ["lambda execution role setup", "s3 bucket policy configuration"]
        vec_docs = [docs[0]]
        vec_distances = [0.2]  # vector_score = 0.8

        retriever, _ = setup_hybrid_search(docs, vec_docs, vec_distances)
        results = retriever.hybrid_search("lambda role", alpha=0.5, top_k=10)

        hit = next(r for r in results if r["text"] == docs[0])
        expected = 0.5 * hit["vector_score"] + 0.5 * hit["bm25_score"]
        assert abs(hit["hybrid_score"] - expected) < 1e-6

    def test_results_sorted_descending_by_hybrid_score(self):
        docs = ["lambda execution role setup", "s3 bucket policy config", "dynamodb table index"]
        vec_docs = [docs[0]]
        vec_distances = [0.1]

        retriever, _ = setup_hybrid_search(docs, vec_docs, vec_distances)
        results = retriever.hybrid_search("lambda", alpha=0.5, top_k=10)

        scores = [r["hybrid_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self):
        docs = [f"document {i} with some content" for i in range(10)]
        vec_docs = docs[:3]
        vec_distances = [0.1, 0.2, 0.3]

        retriever, _ = setup_hybrid_search(docs, vec_docs, vec_distances)
        results = retriever.hybrid_search("document", alpha=0.5, top_k=2)

        assert len(results) == 2

    def test_non_vector_doc_has_zero_vector_score(self):
        docs = ["lambda execution role", "s3 bucket access"]
        # Only docs[0] is in vector results
        vec_docs = [docs[0]]
        vec_distances = [0.1]

        retriever, _ = setup_hybrid_search(docs, vec_docs, vec_distances)
        results = retriever.hybrid_search("lambda", alpha=0.5, top_k=10)

        non_vec = next((r for r in results if r["text"] == docs[1]), None)
        if non_vec:
            assert non_vec["vector_score"] == 0
