"""Evaluate RankerB RAG answers using the standard RAG metrics."""

import sys

from evaluate_rag_rankerA import run_evaluation

DEFAULT_PREDICTIONS = "output/rankerB_rag_answers.json"
DEFAULT_GOLD = "data/rag.json"


def main():
  """Mirror ``evaluate_rag.py`` while defaulting to RankerB outputs."""

  if len(sys.argv) > 1:
    prediction_file = sys.argv[1]
  else:
    prediction_file = DEFAULT_PREDICTIONS

  if len(sys.argv) > 2:
    gold_labels = sys.argv[2]
  else:
    gold_labels = DEFAULT_GOLD

  run_evaluation(prediction_file, gold_labels)


if __name__ == "__main__":
  main()
