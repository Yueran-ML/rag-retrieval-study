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


class TfidfIndex:
  """A lightweight TF-IDF index that supports cosine similarity scoring."""

  def __init__(self, corpus_tokens: List[List[str]]) -> None:
    self.corpus_tokens = corpus_tokens
    self.document_count = len(corpus_tokens)
    self.document_lengths = [len(doc) for doc in corpus_tokens]
    self.term_frequencies: List[Counter[str]] = []
    document_frequencies: Dict[str, int] = {}

    for doc_tokens in corpus_tokens:
      term_frequency = Counter(doc_tokens)
      self.term_frequencies.append(term_frequency)
      for token in term_frequency.keys():
        document_frequencies[token] = document_frequencies.get(token, 0) + 1

    self.inverse_document_frequencies: Dict[str, float] = {}
    for token, doc_freq in document_frequencies.items():
      # Smooth the IDF to avoid divide-by-zero in small corpora.
      self.inverse_document_frequencies[token] = math.log(
          (1 + self.document_count) / (1 + doc_freq)
      ) + 1.0

    self.document_vectors: List[Dict[str, float]] = []
    self.document_norms: List[float] = []

    for idx, term_frequency in enumerate(self.term_frequencies):
      doc_vector: Dict[str, float] = {}
      norm_sq = 0.0
      doc_length = self.document_lengths[idx] or 1
      for token, tf in term_frequency.items():
        idf = self.inverse_document_frequencies[token]
        weight = (tf / doc_length) * idf
        doc_vector[token] = weight
        norm_sq += weight * weight
      norm = math.sqrt(norm_sq)
      if norm == 0.0:
        norm = 1.0
      self.document_vectors.append(doc_vector)
      self.document_norms.append(norm)

  def get_scores(self, query_tokens: List[str]) -> List[float]:
    scores: List[float] = [0.0 for _ in range(self.document_count)]
    if not self.document_count or not query_tokens:
      return scores

    query_tf = Counter(query_tokens)
    query_vector: Dict[str, float] = {}
    norm_sq = 0.0
    query_length = len(query_tokens)

    for token, tf in query_tf.items():
      idf = self.inverse_document_frequencies.get(token)
      if idf is None:
        continue
      weight = (tf / query_length) * idf
      query_vector[token] = weight
      norm_sq += weight * weight

    if not query_vector:
      return scores

    query_norm = math.sqrt(norm_sq)
    if query_norm == 0.0:
      query_norm = 1.0

    for idx, doc_vector in enumerate(self.document_vectors):
      dot_product = 0.0
      for token, q_weight in query_vector.items():
        d_weight = doc_vector.get(token)
        if d_weight is None:
          continue
        dot_product += q_weight * d_weight
      doc_norm = self.document_norms[idx]
      if doc_norm == 0.0:
        continue
      scores[idx] = dot_product / (doc_norm * query_norm)

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
  """Simple tokenizer used for sparse retrieval."""
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


def build_hybrid_corpus(
    documents: List[Dict[str, Any]], chunk_size: int
) -> Dict[str, List[Any]]:
  """Prepare corpus texts, metadata, and tokens for hybrid retrieval."""
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


def _topk_indices(scores: List[float], k: int) -> List[int]:
  if k <= 0:
    return []
  return sorted(
      range(len(scores)),
      key=lambda i: scores[i],
      reverse=True,
  )[:k]


def _normalize_score_map(score_map: Dict[int, float]) -> Dict[int, float]:
  if not score_map:
    return {}
  values = list(score_map.values())
  max_score = max(values)
  min_score = min(values)
  if math.isclose(max_score, min_score):
    baseline = 1.0 if max_score > 0 else 0.0
    return {idx: baseline for idx in score_map}
  scale = max_score - min_score
  return {idx: (score - min_score) / scale for idx, score in score_map.items()}


def gen_stage_hybrid(
    corpus: str,
    queries: str,
    output_name: str,
    topk: int = 10,
    chunk_size: int = 256,
    bm25_weight: float = 0.5,
    candidate_pool_multiplier: float = 2.0,
) -> None:
  """Run a hybrid (BM25 + TF-IDF) retrieval pass and persist results.

  Any reranking should be applied by running ``example_reranker.py`` on
  the saved retrieval output so that all retrieval pipelines share the
  same reranking implementation. ``candidate_pool_multiplier`` controls
  how many combined candidates are considered before truncating to the
  final ``topk`` results, which is useful when BM25 and TF-IDF surface
  different high-quality passages.
  """
  print(f"Preparing to write results to {output_name}. Existing content may be replaced.")

  reader = JSONReader()
  data = reader.load_data(corpus)
  print("Corpus Data")
  print("--------------------------")
  print(data[0])
  print("--------------------------")

  print("Initialising hybrid indexes")
  hybrid_corpus = build_hybrid_corpus(data, chunk_size=chunk_size)
  bm25_index = BM25Okapi(hybrid_corpus["tokens"])
  tfidf_index = TfidfIndex(hybrid_corpus["tokens"])
  print("Hybrid indexes ready ...")

  with open(queries, "r") as file:
    query_data = json.load(file)

  print("Query Data")
  print("--------------------------")
  print(query_data[0])
  print("--------------------------")

  retrieval_save_list: List[Dict[str, Any]] = []
  print("Running Retrieval ...")
  candidate_pool = max(topk, int(topk * candidate_pool_multiplier))
  tfidf_weight = 1.0 - bm25_weight

  for data in tqdm(query_data):
    query = data["query"]
    query_tokens = _tokenize(query)
    bm25_scores = bm25_index.get_scores(query_tokens)
    tfidf_scores = tfidf_index.get_scores(query_tokens)

    bm25_top = _topk_indices(bm25_scores, candidate_pool)
    tfidf_top = _topk_indices(tfidf_scores, candidate_pool)
    candidate_indices = set(bm25_top)
    candidate_indices.update(tfidf_top)

    bm25_candidate_scores = {idx: bm25_scores[idx] for idx in candidate_indices}
    tfidf_candidate_scores = {idx: tfidf_scores[idx] for idx in candidate_indices}

    bm25_norm = _normalize_score_map(bm25_candidate_scores)
    tfidf_norm = _normalize_score_map(tfidf_candidate_scores)

    combined_scores = {}
    normalized_components: Dict[int, Dict[str, float]] = {}

    for idx in candidate_indices:
      bm25_component = bm25_norm.get(idx, 0.0)
      tfidf_component = tfidf_norm.get(idx, 0.0)
      combined_scores[idx] = bm25_weight * bm25_component + tfidf_weight * tfidf_component
      normalized_components[idx] = {
          "bm25": bm25_component,
          "tfidf": tfidf_component,
      }

    top_indices = sorted(
        combined_scores.keys(),
        key=lambda i: combined_scores[i],
        reverse=True,
    )[:candidate_pool]

    retrieval_list: List[Dict[str, Any]] = []

    for idx in top_indices:
      text = hybrid_corpus["texts"][idx]
      metadata = hybrid_corpus["metadata"][idx]
      retrieval_list.append(
          {
              "text": text,
              "score": float(combined_scores[idx]),
              "bm25_score": float(bm25_scores[idx]),
              "tfidf_score": float(tfidf_scores[idx]),
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

  output_name = "output/hybrid-rankerD.json"

  # Reranking is handled centrally by example_reranker.py. After running
  # this retrieval script, invoke the reranker on the generated output if
  # you need re-ordered results.
  gen_stage_hybrid(
      corpus,
      queries,
      output_name,
  )
