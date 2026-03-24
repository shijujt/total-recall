# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Install dependencies (uv recommended)
uv sync

# Ollama must be running with llama3 before any pipeline or eval commands
ollama serve        # separate terminal
ollama pull llama3
```

## Commands

All scripts must be run from `src/`:

```bash
cd src

# Index AWS docs into Chroma (only needed once, or after docs change)
python main_indexer.py

# Run a sample query through the full pipeline
python main_pipeline.py

# Evaluate retrieval quality against eval_queries_ag.jsonl (500 queries)
python main_ret_eval.py
```

No linting, formatting, or test framework is configured in this project.

## Architecture

The system is a multi-stage RAG pipeline for AWS documentation retrieval:

```
Query → ServicePredictor → LlamaQueryRewriter → HybridRetriever → Reranker → Top-K results
```

**Stage details:**

1. **ServicePredictor** (`rag_service_predictor.py`) — cosine similarity between the query embedding and pre-defined service keyword descriptions using `all-MiniLM-L6-v2`. Returns a service name used to filter the Chroma collection.

2. **LlamaQueryRewriter** (`qr/query_writer.py`) — calls Ollama (`http://localhost:11434/api/generate`, model `llama3`) to produce N alternative query formulations. `QueryRewriter` is an abstract base class; `OpenAIQueryRewriter` is an alternative implementation.

3. **HybridRetriever** (`rag_retriever.py`) — runs vector search and BM25 in parallel, unions candidates, and blends scores:
   ```
   hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score   (default alpha=0.5)
   ```
   BM25 index is built on-the-fly from the full Chroma collection at initialization. Service filtering uses a Chroma `where={"service": <name>}` clause.

4. **Reranker** (`rag_reranker.py`) — scores the union of candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`. This is the final ranking signal.

**Indexing pipeline** (`rag_indexer.py`):
- `MarkdownSectionParser` splits docs with a 3-tier fallback: H2 headings → bold sections → fixed token windows (400 tokens)
- `AwsSvcIndexer` writes chunks to Chroma at `../chroma_db/` (relative to `src/`), collection `"aws_docs"`
- Each document stored with metadata: `service`, `file`, `section_title`, `heading_level`

**Evaluation** (`rag_ret_eval.py`):
- Loads JSONL queries with fields: `query`, `service`, `src_doc`, `keywords`
- Relevance = keyword substring match against retrieved chunk text
- Reports Recall@k, HitRate@1, MRR
