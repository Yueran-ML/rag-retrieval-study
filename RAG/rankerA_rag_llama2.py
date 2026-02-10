import json, os, re
import psutil
import torch
from typing import Any, Generator, List, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from llama_index.core.schema import Document


GPU = True
STAGING = False

if GPU:
    torch.set_default_dtype(torch.float16)
else:
    device = torch.device('cpu')

def save_list_to_json(lst, filename):
    with open(filename, 'w') as file:
        json.dump(lst, file)

def wr_dict(filename, dic):
    try:
        if not os.path.isfile(filename):
            data = [dic]
            with open(filename, 'w') as f:
                json.dump(data, f)
        else:
            with open(filename, 'r') as f:
                data = json.load(f)
                data.append(dic)
            with open(filename, 'w') as f:
                json.dump(data, f)
    except Exception as e:
        print("Save Error:", str(e))
    return

def rm_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} removed successfully.")

def _depth_first_yield(json_data: Any, levels_back: int, collapse_length: Optional[int], path: List[str], ensure_ascii: bool = False) -> Generator[str, None, None]:
    if isinstance(json_data, (dict, list)):
        json_str = json.dumps(json_data, ensure_ascii=ensure_ascii)
        if collapse_length is not None and len(json_str) <= collapse_length:
            new_path = path[-levels_back:]
            new_path.append(json_str)
            yield " ".join(new_path)
            return
        elif isinstance(json_data, dict):
            for key, value in json_data.items():
                new_path = path[:]
                new_path.append(key)
                yield from _depth_first_yield(value, levels_back, collapse_length, new_path)
        elif isinstance(json_data, list):
            for _, value in enumerate(json_data):
                yield from _depth_first_yield(value, levels_back, collapse_length, path)
        else:
            new_path = path[-levels_back:]
            new_path.append(str(json_data))
            yield " ".join(new_path)

class JSONReader():
    def __init__(self, is_jsonl: Optional[bool] = False,) -> None:
        super().__init__()
        self.is_jsonl = is_jsonl
    def load_data(self, input_file: str) -> List[Document]:
        documents = []
        with open(input_file, 'r') as file:
            load_data = json.load(file)
        for data in load_data:
            metadata = {"title": data['title'], "published_at": data['published_at'], "source": data['source']}
            documents.append(Document(text=data['body'], metadata=metadata))
        return documents

BASE_PREFIX = """Below is a question followed by some context from different sources. 
          Please answer the question based on the context. 
          The answer to the question is a word or entity. 
          If the provided information is insufficient to answer the question, 
          respond 'Insufficient Information'. 
          Answer directly without explanation."""

def postprocess_freeform(ans: str) -> str:
    a = ans.strip()
    a = re.sub(r"^answer\s*[:\-–]\s*", "", a, flags=re.I)
    a = re.sub(r"^the answer is\s*[:\-–]?\s*", "", a, flags=re.I)
    a = a.replace("\n", " ").strip()
    a = re.sub(r"\s+", " ", a)
    a = re.sub(r'^[\'"`]+|[\'"`]+$', '', a)
    a = re.sub(r"[\.。!\?]\s*$", "", a)
    return a.strip()

def normalize_comparison(ans: str) -> str:
    m = re.search(r'\b(yes|no|true|false|agree|disagree|same|different)\b', ans, re.I)
    if m:
        tok = m.group(1).lower()
        table = {'yes': 'Yes', 'no': 'No', 'true': 'True', 'false': 'False', 'agree': 'Agree', 'disagree': 'Disagree', 'same': 'Same', 'different': 'Different'}
        return table[tok]
    return postprocess_freeform(ans).capitalize()

def normalize_null(ans: str) -> str:
    if re.search(r'insufficient\s+information', ans, re.I) or re.search(r'not\s+enough\s+information|unknown|cannot\s+answer|no\s+evidence', ans, re.I):
        return 'Insufficient Information'
    return postprocess_freeform(ans)

def run_query(tokenizer, model, messages, temperature=None, max_new_tokens=512, **kwargs):
    messages = [{"role": "user", "content": messages}]
    ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    if GPU:
        ids = ids.cuda()
        attention_mask = torch.ones_like(ids)
    do_sample = kwargs.pop("do_sample", False)
    gen_temp = kwargs.pop("temperature", temperature)
    if do_sample:
        generation_config = GenerationConfig(do_sample=True, temperature=gen_temp, **kwargs)
    else:
        generation_config = GenerationConfig(do_sample=False)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = out.sequences[0]
    gen_only = s[ids.shape[-1]:]
    text = tokenizer.decode(gen_only, skip_special_tokens=True)
    return text.strip()


def initialise_and_run_model(save_name, input_stage_1, model_name):
    if GPU:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cpu")

    print('Loading Stage 1 Ranking')
    with open(input_stage_1, 'r') as file:
        doc_data = json.load(file)

    print('Remove saved file if exists.')
    rm_file(save_name)

    save_list = []
    for d in tqdm(doc_data):
        retrieval_list = d['retrieval_list']
        context = '--------------'.join(e['text'] for e in retrieval_list)

        if d['question_type'] == 'comparison_query':
            comp_prefix = (
                "Answer ONLY ONE token from this set:\n"
                "Yes | No | True | False | Agree | Disagree | Same | Different\n"
                "Do NOT include explanations or punctuation."
            )
            prompt = f"{comp_prefix}\n\nQuestion:{d['query']}\n\nContext:\n\n{context}"
            response = run_query(tokenizer, model, prompt, do_sample=False, temperature=0.0, max_new_tokens=8)
            resp = normalize_comparison(response)
        else:
            prompt = f"{BASE_PREFIX}\n\nQuestion:{d['query']}\n\nContext:\n\n{context}"
            response = run_query(tokenizer, model, prompt, do_sample=False, temperature=0.0, max_new_tokens=32)
            if d['question_type'] == 'null_query':
                resp = normalize_null(response)
            else:
                resp = postprocess_freeform(response)

        qid = d.get("question_id") or d.get("id") or d.get("qid")
        save = {
            "question_id": qid,
            "query": d["query"],
            "prompt": prompt,
            "predicted_answer": resp,
            "model_answer": resp,
            "gold_answer": d["answer"],
            "question_type": d["question_type"],
        }
        save_list.append(save)

    print('Query processing completed. Saving the results.')
    save_list_to_json(save_list, save_name)

if __name__ == '__main__':
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    output_file = "output/llama2-rankerA.json"
    input_stage_1 = 'output/embedder-rankerA.json'
    initialise_and_run_model(output_file, input_stage_1, model_name)

    if STAGING:
        process = psutil.Process(os.getpid())
        peak_wset_bytes = process.memory_info().rss
        peak_wset_gb = peak_wset_bytes / (1024 * 1024 * 1024)
        print(f"Peak working set size: {peak_wset_gb:.2f} GB")
