# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Install all dependencies including dev tools
uv sync --group dev

# Ollama must be running with llama3 before any pipeline or eval commands
ollama serve        # separate terminal
ollama pull llama3
```

## Commands

Run from project root:

```bash
# Linting and formatting
uv run ruff check src tests   # lint
uv run ruff format src tests  # format
uv run mypy src               # type check

# Tests
uv run pytest                 # run all tests with coverage

# Pipeline scripts (require Ollama running + chroma_db populated)
uv run python scripts/run_indexer.py   # index AWS docs into Chroma
uv run python scripts/run_pipeline.py  # run a sample query
uv run python scripts/run_eval.py      # evaluate retrieval quality
```

## Architecture

The system is a multi-stage RAG pipeline for AWS documentation retrieval:

```
Query → ServicePredictor → LlamaQueryRewriter → HybridRetriever → Reranker → Top-K results
```

**Stage details:**

1. **ServicePredictor** (`src/ir/service_predictor.py`) — cosine similarity between the query embedding and pre-defined service keyword descriptions using `all-MiniLM-L6-v2`. Returns a service name used to filter the Chroma collection.

2. **LlamaQueryRewriter** (`src/ir/qr/query_writer.py`) — calls Ollama (`http://localhost:11434/api/generate`, model `llama3`) to produce N alternative query formulations. `QueryRewriter` is an abstract base class; `OpenAIQueryRewriter` is an alternative implementation.

3. **HybridRetriever** (`src/ir/retriever.py`) — runs vector search and BM25 in parallel, unions candidates, and blends scores:
   ```
   hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score   (default alpha=0.5)
   ```
   BM25 index is built on-the-fly from the full Chroma collection at initialization. Service filtering uses a Chroma `where={"service": <name>}` clause.

4. **Reranker** (`src/ir/reranker.py`) — scores the union of candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`. This is the final ranking signal.

**Indexing pipeline** (`src/ir/indexer.py`):
- `MarkdownSectionParser` splits docs with a 3-tier fallback: H2 headings → bold sections → fixed token windows (400 tokens)
- `AwsSvcIndexer` writes chunks to Chroma at `chroma_db/` (project root), collection `"aws_docs"`
- Each document stored with metadata: `service`, `file`, `section_title`, `heading_level`, `chunk_id`
- Dual-mode: pass `collection_name="aws_docs"` to index, `collection_name=None` to generate eval queries

**Evaluation** (`src/ir/evaluator.py`):
- Loads JSONL queries with fields: `query`, `service`, `src_doc`, `keywords`
- Relevance = keyword substring match against retrieved chunk text
- Reports Recall@k, HitRate@1, MRR
