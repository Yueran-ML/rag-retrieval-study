# NLP 

This document explains the naming convention and purpose of each file in this RAG (Retrieval-Augmented Generation) system.

## File Naming Convention

### Retrieval Files (Rankers)

| File | Algorithm | Output File | Description |
|------|-----------|-------------|-------------|
| `rankerA.py` | Dense Vector Retrieval | `embedder-rankerA.json` | Uses embedding models for semantic search |
| `rankerB.py` | BM25 | `bm25-rankerB.json` | Classic sparse retrieval algorithm |
| `rankerC.py` | TF-IDF | `tfidf-rankerC.json` | Term frequency-inverse document frequency |
| `rankerD.py` | Hybrid (BM25+TF-IDF) | `hybrid-rankerD.json` | Combines BM25 and TF-IDF scores |

### RAG Generation Files

**Format**: `{ranker}_{rag}_{model}.py` → `{model}-{ranker}.json`

#### RankerA Series (Dense Retrieval)
| File | LLM Model | Output File |
|------|-----------|-------------|
| `rankerA_rag_llama2.py` | Llama-2-7b-chat-hf | `llama2-rankerA.json` |
| `rankerA_rag_llama3.py` | Meta-Llama-3-8B-Instruct | `llama3-rankerA.json` |
| `rankerA_rag_mistral.py` | Mistral-7B-Instruct-v0.2 | `mistral-rankerA.json` |
| `rankerA_rag_qwen2.py` | Qwen2-7B-Instruct | `qwen2-rankerA.json` |

#### RankerB Series (BM25 Retrieval)
| File | LLM Model | Output File |
|------|-----------|-------------|
| `rankerB_rag_llama2.py` | Llama-2-7b-chat-hf | `llama2-rankerB.json` |
| `rankerB_rag_llama3.py` | Meta-Llama-3-8B-Instruct | `llama3-rankerB.json` |
| `rankerB_rag_mistral.py` | Mistral-7B-Instruct-v0.2 | `mistral-rankerB.json` |
| `rankerB_rag_qwen2.py` | Qwen2-7B-Instruct | `qwen2-rankerB.json` |

## Data Files

| File | Purpose |
|------|---------|
| `data/corpus.json` | Full document corpus |
| `data/rag.json` | Full query and answer set |
| `data/sample-corpus.json` | Sample corpus for testing |
| `data/sample-rag.json` | Sample queries for testing |

## Output Files

All output files are stored in the `output/` directory:

### Retrieval Results
- `embedder-rankerA.json` - Dense vector retrieval results
- `bm25-rankerB.json` - BM25 retrieval results  
- `tfidf-rankerC.json` - TF-IDF retrieval results
- `hybrid-rankerD.json` - Hybrid retrieval results

### Generated Answers
- `llama2-rankerA.json` - Llama2 answers using dense retrieval
- `llama2-rankerB.json` - Llama2 answers using BM25 retrieval
- `llama3-rankerA.json` - Llama3 answers using dense retrieval
- `llama3-rankerB.json` - Llama3 answers using BM25 retrieval
- `mistral-rankerA.json` - Mistral answers using dense retrieval
- `mistral-rankerB.json` - Mistral answers using BM25 retrieval
- `qwen2-rankerA.json` - Qwen2 answers using dense retrieval
- `qwen2-rankerB.json` - Qwen2 answers using BM25 retrieval

