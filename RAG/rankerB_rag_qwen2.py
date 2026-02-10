"""Example script for scoring rankerB retrieval outputs with the Qwen2-7B-Instruct model."""
import os
import psutil

from rankerA_rag_llama2 import STAGING, initialise_and_run_model


if __name__ == "__main__":
    model_name = "Qwen/Qwen2-7B-Instruct"
    output_file = "output/qwen2-rankerB.json"
    input_stage_1 = "output/bm25-rankerB.json"

    initialise_and_run_model(output_file, input_stage_1, model_name)

    if STAGING:
        process = psutil.Process(os.getpid())
        peak_wset_bytes = process.memory_info().rss
        peak_wset_gb = peak_wset_bytes / (1024 * 1024 * 1024)
        print(f"Peak working set size: {peak_wset_gb:.2f} GB")