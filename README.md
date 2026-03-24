# AWS Documentation RAG System

A Retrieval-Augmented Generation (RAG) pipeline for answering AWS developer questions by searching and ranking documentation across 24+ AWS services.

## Overview

The system indexes AWS service documentation into a vector database and retrieves relevant sections through a multi-stage pipeline:

1. **Service Prediction** — identifies which AWS service a query targets
2. **Query Rewriting** — generates alternative formulations to improve recall
3. **Hybrid Retrieval** — combines semantic vector search with BM25 keyword matching
4. **Reranking** — scores candidates with a cross-encoder model for final ordering

## Architecture

```
User Query
    │
    ▼
Service Predictor       ← cosine similarity against service embeddings
    │  (service filter)
    ▼
Query Rewriter          ← Llama3 via Ollama generates N alternative queries
    │  (expanded queries)
    ▼
Hybrid Retriever        ← vector search + BM25, union of candidates
    │  (deduplicated candidates)
    ▼
Reranker                ← CrossEncoder scores each candidate
    │
    ▼
Top-K Results           ← with vector, BM25, hybrid, and rerank scores
```

### Component Summary

| Component | File | Description |
|---|---|---|
| `RAGPipeline` | `rag_pipeline.py` | Orchestrates the full pipeline |
| `HybridRetriever` | `rag_retriever.py` | Vector + BM25 hybrid search |
| `Reranker` | `rag_reranker.py` | CrossEncoder reranking |
| `ServicePredictor` | `rag_service_predictor.py` | Identifies target AWS service |
| `LlamaQueryRewriter` | `qr/query_writer.py` | Query expansion via Llama3 |
| `AwsSvcIndexer` | `rag_indexer.py` | Parses and indexes documentation |
| `RetrievalEvaluator` | `rag_ret_eval.py` | Evaluation metrics (Recall, MRR) |

## Prerequisites

- Python 3.12
- [Ollama](https://ollama.com/) running locally with the `llama3` model

```bash
ollama serve          # in a separate terminal
ollama pull llama3
```

## Installation

```bash
git clone <repo>
cd ir

python -m venv .venv
source .venv/bin/activate

# using uv (recommended)
pip install uv
uv sync

# or pip directly
pip install chromadb sentence-transformers torch rank-bm25 nltk numpy requests
```

## Usage

### 1. Index AWS Documentation

Parses markdown files from `docs/` and stores chunks in a local Chroma database (`chroma_db/`).

```bash
cd src
python main_indexer.py
```

### 2. Query the Pipeline

Runs example queries against the indexed collection and displays ranked results with scores.

```bash
cd src
python main_pipeline.py
```

Example output:
```
Query: "how to set up basic execution role for lambda function"
Service: lambda (confidence: 0.87)

Rank 1 — rerank: 9.43 | hybrid: 0.81 | vector: 0.79 | bm25: 0.83
  [lambda] permissions/execution-role.md — Setting Up Execution Roles
  ...
```

### 3. Run Evaluation

Evaluates retrieval quality against `eval_queries_ag.jsonl` (500 synthetic queries).

```bash
cd src
python main_ret_eval.py
```

Reports Recall@k, HitRate@1, and MRR.

## Configuration

### RAGPipeline

| Parameter | Default | Description |
|---|---|---|
| `top_k` | `5` | Number of final results to return |

### HybridRetriever

| Parameter | Default | Description |
|---|---|---|
| `alpha` | `0.5` | Blend weight: `0` = pure vector, `1` = pure BM25 |
| `vector_top_k` | `30` | Vector search candidate count |
| `bm25_top_k` | `30` | BM25 candidate count |
| `service_filter` | `None` | Restrict search to a specific AWS service |

### LlamaQueryRewriter

| Parameter | Default | Description |
|---|---|---|
| `n_queries` | `2` | Number of alternative query formulations |
| `model` | `llama3` | Ollama model to use |
| `endpoint` | `http://localhost:11434/api/generate` | Ollama API endpoint |

### MarkdownSectionParser

| Parameter | Default | Description |
|---|---|---|
| `max_tokens` | `400` | Token window for fallback splitting |

### Models Used

| Model | Source | Purpose |
|---|---|---|
| `all-MiniLM-L6-v2` | Hugging Face | Document + query embeddings |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face | Reranking |
| `llama3` | Ollama (local) | Query rewriting |

## Evaluation

Test queries are stored in JSONL format:

```json
{
  "query": "How do I specify a custom runtime for AWS Lambda?",
  "service": "lambda",
  "src_doc": "runtimes-custom.md",
  "keywords": ["using a custom runtime"]
}
```

Relevance is determined by keyword matching against retrieved document text.

**Metrics:**
- **Recall@k** — fraction of queries with a relevant result in the top-k
- **HitRate@1** — fraction with a relevant result at rank 1
- **MRR** — Mean Reciprocal Rank across all queries

## Project Structure

```
ir/
├── src/
│   ├── main_indexer.py           # Index documentation
│   ├── main_pipeline.py          # Run example queries
│   ├── main_ret_eval.py          # Run evaluation
│   ├── rag_pipeline.py           # Pipeline orchestration
│   ├── rag_retriever.py          # Hybrid retrieval (vector + BM25)
│   ├── rag_reranker.py           # CrossEncoder reranking
│   ├── rag_indexer.py            # Document parsing and indexing
│   ├── rag_ret_eval.py           # Evaluation harness
│   ├── rag_service_predictor.py  # AWS service classification
│   └── qr/
│       └── query_writer.py       # Query rewriting (Llama, OpenAI)
├── docs/
│   ├── classes.mmd               # Class diagram
│   ├── packages.mmd              # Module dependency diagram
│   ├── seq_indexer.mmd           # Sequence diagram: main_indexer
│   ├── seq_pipeline.mmd          # Sequence diagram: main_pipeline
│   └── seq_eval.mmd              # Sequence diagram: main_ret_eval
├── chroma_db/                    # Persistent vector store (auto-created)
├── eval_queries.jsonl            # Evaluation queries (manual)
├── eval_queries_ag.jsonl         # Evaluation queries (Llama-generated, 500)
└── pyproject.toml
```

## Supported AWS Services

S3, Lambda, DynamoDB, SNS, SQS, Step Functions, API Gateway, Glue, IAM, CloudFormation, AWS CLI, and more.
