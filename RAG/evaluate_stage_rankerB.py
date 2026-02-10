"""Evaluate BM25 retrieval results using the stage 1 metrics."""

from evaluate_stage_rankerA import main_eval


if __name__ == "__main__":
  main_eval("output/bm25-ranker.json")
