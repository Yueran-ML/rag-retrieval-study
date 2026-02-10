import os, psutil
from rankerA_rag_llama2 import STAGING, initialise_and_run_model

if __name__ == "__main__":
    model_name = "Qwen/Qwen2-7B-Instruct"
    input_stage_1 = os.environ.get("RAG_INPUT", "output/embedder-rankerA.json")
    output_file = os.environ.get("RAG_OUTPUT", "output/qwen2-rankerA.json")
    initialise_and_run_model(output_file, input_stage_1, model_name)
    if STAGING:
        g = psutil.Process().memory_info().rss/(1024**3)
        print(f"Peak working set size: {g:.2f} GB")

