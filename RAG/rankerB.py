import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, Generator, List, Optional

from tqdm import tqdm

STAGING = False


class BM25Okapi:
  """A lightweight BM25 implementation with the same interface we use."""

  def __init__(
      self,
      corpus_tokens: List[List[str]],
      k1: float = 1.5,
      b: float = 0.75,
  ) -> None:
    self.k1 = k1
    self.b = b
    self.corpus_tokens = corpus_tokens
    self.document_count = len(corpus_tokens)
    self.document_lengths = [len(doc) for doc in corpus_tokens]
    self.avg_document_length = (
        sum(self.document_lengths) / self.document_count
        if self.document_count
        else 0.0
    )
    if self.avg_document_length == 0:
      self.avg_document_length = 1.0

    self.term_frequencies: List[Counter[str]] = []
    document_frequencies: Dict[str, int] = {}

    for doc_tokens in corpus_tokens:
      term_frequency = Counter(doc_tokens)
      self.term_frequencies.append(term_frequency)
      for token in term_frequency.keys():
        document_frequencies[token] = document_frequencies.get(token, 0) + 1

    self.inverse_document_frequencies: Dict[str, float] = {}
    for token, doc_freq in document_frequencies.items():
      # Standard BM25 IDF with +1 smoothing to avoid division-by-zero.
      self.inverse_document_frequencies[token] = math.log(
          1 + ((self.document_count - doc_freq + 0.5) / (doc_freq + 0.5))
      )

  def get_scores(self, query_tokens: List[str]) -> List[float]:
    scores: List[float] = [0.0 for _ in range(self.document_count)]
    if not self.document_count:
      return scores

    for token in query_tokens:
      idf = self.inverse_document_frequencies.get(token)
      if idf is None:
        continue

      for idx, term_frequency in enumerate(self.term_frequencies):
        tf = term_frequency.get(token)
        if tf is None:
          continue

        doc_length = self.document_lengths[idx]
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (
            1 - self.b + self.b * (doc_length / self.avg_document_length)
        )
        scores[idx] += idf * (numerator / denominator)

    return scores

def save_list_to_json(lst: List[Dict[str, Any]], filename: str) -> None:
  """Persist a list of dictionaries to a JSON file."""
  with open(filename, "w") as file:
    json.dump(lst, file)


def wr_dict(filename: str, dic: Dict[str, Any]) -> None:
  """Append a dictionary to a JSON file, creating the file if required."""
  try:
    if not os.path.isfile(filename):
      data = [dic]
      with open(filename, "w") as f:
        json.dump(data, f)
    else:
      with open(filename, "r") as f:
        data = json.load(f)
        data.append(dic)
      with open(filename, "w") as f:
        json.dump(data, f)
  except Exception as e:
    print("Save Error:", str(e))


def _depth_first_yield(
    json_data: Any,
    levels_back: int,
    collapse_length: Optional[int],
    path: List[str],
    ensure_ascii: bool = False,
) -> Generator[str, None, None]:
  """Depth first traversal helper used for inspecting JSON structures."""
  if isinstance(json_data, (dict, list)):
    json_str = json.dumps(json_data, ensure_ascii=ensure_ascii)
    if collapse_length is not None and len(json_str) <= collapse_length:
      new_path = path[-levels_back:]
      new_path.append(json_str)
      yield " ".join(new_path)
      return
    if isinstance(json_data, dict):
      for key, value in json_data.items():
        new_path = path[:]
        new_path.append(key)
        yield from _depth_first_yield(value, levels_back, collapse_length, new_path)
    elif isinstance(json_data, list):
      for value in json_data:
        yield from _depth_first_yield(value, levels_back, collapse_length, path)
  else:
    new_path = path[-levels_back:]
    new_path.append(str(json_data))
    yield " ".join(new_path)


class JSONReader:
  """Utility for reading corpus documents stored as JSON."""

  def __init__(self, is_jsonl: Optional[bool] = False) -> None:
    super().__init__()
    self.is_jsonl = is_jsonl

  def load_data(self, input_file: str) -> List[Dict[str, Any]]:
    documents = []
    with open(input_file, "r") as file:
      load_data = json.load(file)
    for data in load_data:
      metadata = {
        "title": data["title"],
        "published_at": data["published_at"],
        "source": data["source"],
      }
      documents.append({"text": data["body"], "metadata": metadata})
    return documents


def _tokenize(text: str) -> List[str]:
  """Simple tokenizer used for BM25."""
  return re.findall(r"\b\w+\b", text.lower())


def _split_document(text: str, chunk_size: int = 256) -> List[str]:
  """Split documents into sentence-like chunks to mirror dense retrieval."""
  sentences = re.split(r"(?<=[.!?])\s+", text)
  chunks: List[str] = []
  current_chunk: List[str] = []
  current_length = 0

  for sentence in sentences:
    if not sentence:
      continue
    sentence_length = len(sentence)
    if current_length + sentence_length > chunk_size and current_chunk:
      chunks.append(" ".join(current_chunk).strip())
      current_chunk = [sentence]
      current_length = sentence_length
    else:
      current_chunk.append(sentence)
      current_length += sentence_length

  if current_chunk:
    chunks.append(" ".join(current_chunk).strip())

  return [chunk for chunk in chunks if chunk]


def build_bm25_corpus(
    documents: List[Dict[str, Any]], chunk_size: int
) -> Dict[str, List[Any]]:
  """Prepare corpus texts, metadata, and tokens for BM25 retrieval."""
  corpus_texts: List[str] = []
  corpus_metadata: List[Dict[str, Any]] = []
  tokenized_corpus: List[List[str]] = []

  for doc in documents:
    chunks = _split_document(doc["text"], chunk_size=chunk_size)
    if not chunks:
      chunks = [doc["text"]]
    for idx, chunk in enumerate(chunks):
      corpus_texts.append(chunk)
      metadata = dict(doc["metadata"])
      metadata["chunk_id"] = idx
      corpus_metadata.append(metadata)
      tokenized_corpus.append(_tokenize(chunk))

  return {
    "texts": corpus_texts,
    "metadata": corpus_metadata,
    "tokens": tokenized_corpus,
  }


def gen_stage_bm25(
    corpus: str,
    queries: str,
    output_name: str,
    topk: int = 10,
    chunk_size: int = 256,
) -> None:
  """Run a BM25 retrieval pass and persist the ranked results.

  Any reranking should be applied by running ``example_reranker.py`` on
  the saved retrieval output so that all retrieval pipelines share the
  same reranking implementation.
  """
  print(f"Preparing to write results to {output_name}. Existing content may be replaced.")

  reader = JSONReader()
  data = reader.load_data(corpus)
  print("Corpus Data")
  print("--------------------------")
  print(data[0])
  print("--------------------------")

  print("Initialising BM25 index")
  bm25_corpus = build_bm25_corpus(data, chunk_size=chunk_size)
  bm25 = BM25Okapi(bm25_corpus["tokens"])
  print("BM25 index ready ...")

  with open(queries, "r") as file:
    query_data = json.load(file)

  print("Query Data")
  print("--------------------------")
  print(query_data[0])
  print("--------------------------")

  retrieval_save_list: List[Dict[str, Any]] = []
  print("Running Retrieval ...")
  candidate_pool = topk

  for data in tqdm(query_data):
    query = data["query"]
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)
    sorted_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )
    top_indices = sorted_indices[:candidate_pool]

    retrieval_list: List[Dict[str, Any]] = []

    for idx in top_indices:
      text = bm25_corpus["texts"][idx]
      metadata = bm25_corpus["metadata"][idx]
      retrieval_list.append(
          {
              "text": text,
              "score": float(scores[idx]),
              "metadata": metadata,
          }
      )

    save: Dict[str, Any] = {}
    save["query"] = data["query"]
    save["answer"] = data["answer"]
    save["question_type"] = data["question_type"]
    save["retrieval_list"] = retrieval_list
    save["gold_list"] = data["evidence_list"]
    retrieval_save_list.append(save)

  print(f"Retrieval complete. Writing results to {output_name}")
  save_list_to_json(retrieval_save_list, output_name)


if __name__ == "__main__":
  if STAGING:
    corpus = "data/sample-corpus.json"
    queries = "data/sample-rag.json"
  else:
    corpus = "data/corpus.json"
    queries = "data/rag.json"

  output_name = "output/bm25-rankerB.json"
  rerank = True

  # Reranking is handled centrally by example_reranker.py. After running
  # this retrieval script, invoke the reranker on the generated output if
  # you need re-ordered results.
  gen_stage_bm25(
      corpus,
      queries,
      output_name,
  )
