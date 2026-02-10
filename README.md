# First-Stage Retrieval Strategies for RAG: A Comparative Study

An experimental study comparing four first-stage retrieval methods and their interaction with instruction-tuned LLMs in a Retrieval-Augmented Generation (RAG) pipeline. All experiments use deterministic decoding, fixed chunk size, and no reranking, isolating the effect of retrieval quality on downstream answer generation.

## Table of Contents

- [Research Questions](#research-questions)
- [Key Findings](#key-findings)
- [Experimental Setup](#experimental-setup)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Report](#report)
- [License](#license)

## Research Questions

1. **How do sparse, dense, and hybrid first-stage retrievers compare** on Hits@k, MRR@10, and MAP@10 when chunk size and top-k are held constant?
2. **To what extent does retrieval method choice propagate** to end-to-end answer quality (token-level Precision, Recall, F1) across different LLMs?
3. **Do instruction-tuned LLMs exhibit differential sensitivity** to retrieval noise introduced by different retriever families?

## Key Findings

- Dense retrieval (BAAI/llm-embedder) achieves the highest stage-1 recall but does not uniformly produce the best end-to-end F1 across all LLMs.
- BM25 remains competitive on factoid-style queries where lexical overlap with the gold evidence is high.
- Hybrid fusion (BM25 + TF-IDF) does not consistently outperform either component retriever in isolation.
- Answer quality varies across LLMs even when provided identical retrieved context, suggesting model-specific prompt sensitivity.

> [!NOTE]
> These findings are based on a single corpus and query set. See [Limitations](#limitations) for caveats.

## Experimental Setup

| Parameter | Value |
|---|---|
| Chunk size | 256 tokens |
| Top-k | 10 |
| Reranking | None |
| Decoding | Greedy (temperature = 0, `do_sample=False`) |
| Corpus format | JSON (title, body, source, published_at) |

### Retrieval Methods

| ID | Method | Implementation |
|---|---|---|
| Ranker A | Dense Embedding | `BAAI/llm-embedder` via LlamaIndex `VectorStoreIndex` |
| Ranker B | BM25 (Okapi) | Custom `BM25Okapi` class (k1=1.5, b=0.75) |
| Ranker C | TF-IDF | Custom `TfidfIndex` class with cosine similarity scoring |
| Ranker D | Hybrid | Min-max normalized BM25 + TF-IDF score fusion (α=0.5) |

### Generator Models

| Model | HuggingFace ID |
|---|---|
| Llama-2-7B | `meta-llama/Llama-2-7b-chat-hf` |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.2` |
| Qwen2-7B | `Qwen/Qwen2-7B-Instruct` |

### Evaluation Metrics

- **Stage-1 retrieval**: Hits@4, Hits@10, MRR@10, MAP@10
- **End-to-end generation**: Token-level Precision, Recall, F1 (macro-averaged across queries)

## Repository Structure

```
├── README.md
├── report.pdf                         # Full experimental report
└── RAG/
    ├── data/
    │   ├── corpus.json                # Document corpus
    │   ├── rag.json                   # Query-answer pairs with gold evidence
    │   ├── sample-corpus.json         # Subset for local testing
    │   └── sample-rag.json            # Subset for local testing
    ├── output/                        # Pre-computed retrieval and generation outputs
    │   ├── embedder-rankerA.json      # Dense retrieval results
    │   ├── bm25-rankerB.json          # BM25 retrieval results
    │   ├── tfidf-rankerC.json         # TF-IDF retrieval results
    │   ├── hybrid-rankerD.json        # Hybrid retrieval results
    │   ├── llama2-rankerA.json        # Llama-2 answers (dense retrieval)
    │   ├── llama3-rankerA.json        # Llama-3 answers (dense retrieval)
    │   ├── mistral-rankerA.json       # Mistral answers (dense retrieval)
    │   ├── qwen2-rankerA.json         # Qwen2 answers (dense retrieval)
    │   └── ...                        # Additional model × retriever combinations
    ├── rankerA.py                     # Dense retrieval pipeline
    ├── rankerB.py                     # BM25 retrieval pipeline
    ├── rankerC.py                     # TF-IDF retrieval pipeline
    ├── rankerD.py                     # Hybrid retrieval pipeline
    ├── rankerA_rag_llama2.py          # RAG generation: Llama-2 + dense retrieval
    ├── rankerA_rag_llama3.py          # RAG generation: Llama-3 + dense retrieval
    ├── rankerA_rag_mistral.py         # RAG generation: Mistral + dense retrieval
    ├── rankerA_rag_qwen2.py           # RAG generation: Qwen2 + dense retrieval
    ├── rankerB_rag_*.py               # RAG generation scripts for BM25
    ├── evaluate_stage_rankerA.py      # Stage-1 retrieval evaluation
    ├── evaluate_rag_rankerA.py        # End-to-end answer evaluation
    ├── example_reranker.py            # Optional reranker reference implementation
    ├── requirements.txt               # Python dependencies
    └── test_runner.sh                 # Batch execution script
```

## Reproducibility

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (≥16 GB VRAM recommended for 7B/8B models)
- HuggingFace access tokens for gated models (Llama-2, Llama-3)

### Installation

```bash
cd RAG
pip install -r requirements.txt
```

### Running Retrieval (Stage 1)

Each retriever can be executed independently. Results are written to `output/`.

```bash
# Dense retrieval (BAAI/llm-embedder)
python rankerA.py

# BM25
python rankerB.py

# TF-IDF
python rankerC.py

# Hybrid (BM25 + TF-IDF)
python rankerD.py
```

### Running Generation (Stage 2)

Generation scripts consume the stage-1 output and produce answer files.

```bash
# Example: Llama-2 with dense retrieval context
python rankerA_rag_llama2.py

# Example: Mistral with BM25 retrieval context
python rankerB_rag_mistral.py
```

### Local Testing

Set `STAGING = True` in any script to run against the sample data (`data/sample-corpus.json`, `data/sample-rag.json`) for quick iteration without GPU.

## Evaluation

### Stage-1 Retrieval Metrics

```bash
python evaluate_stage_rankerA.py
```

Reports Hits@4, Hits@10, MRR@10, and MAP@10. Modify the `stage_one_filename` variable to evaluate other retrievers.

### End-to-End Answer Metrics

```bash
python evaluate_rag_rankerA.py output/llama2-rankerA.json
```

Reports per-question-type and overall Precision, Recall, and F1. Accepts any generation output file as argument.

### Pre-computed Outputs

All retrieval and generation outputs are included under `RAG/output/` for direct evaluation without re-running the full pipeline.

## Limitations

- **Single dataset**: All experiments use one corpus and query set; generalisability to other domains is not established.
- **No reranking**: The pipeline omits a reranking stage, which would likely improve retrieval precision for all methods.
- **Fixed hyperparameters**: Chunk size (256), top-k (10), and BM25 parameters (k1=1.5, b=0.75) are not tuned.
- **Greedy decoding only**: Sampling-based decoding strategies are not explored.
- **Limited model scale**: All generators are 7B–8B parameter models; larger models may exhibit different retrieval sensitivity.
- **No statistical significance testing**: Results are from single runs without confidence intervals.

## Report

The full experimental report, including detailed results tables and analysis, is available in [`report.pdf`](report.pdf).

## License

This repository is provided for research and educational purposes.
