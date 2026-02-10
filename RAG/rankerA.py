import json, os
from tqdm import tqdm
from copy import deepcopy
from typing import Any, Generator, List, Dict, Optional
import psutil
import openai

# 0.10.x 的正确导入
from llama_index.core import (
    ServiceContext,
    PromptHelper,
    VectorStoreIndex,
    set_global_service_context,
)
from llama_index.core.extractors import BaseExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.schema import QueryBundle, MetadataMode, Document

# LLM / Embeddings 分包
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding
# 用 Cohere 时（可选，装了分包才能用）：
# from llama_index.embeddings.cohereai import CohereEmbedding

# FlagEmbedding 重排器（路径变了）
try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    HAVE_FLAG_RERANKER = True
except Exception:
    HAVE_FLAG_RERANKER = False


STAGING = False


def save_list_to_json(lst, filename):
    with open(filename, "w") as file:
        json.dump(lst, file)


def wr_dict(filename, dic):
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
    return


def _depth_first_yield(
    json_data: Any,
    levels_back: int,
    collapse_length: Optional[int],
    path: List[str],
    ensure_ascii: bool = False,
) -> Generator[str, None, None]:
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


class JSONReader:
    def __init__(self, is_jsonl: Optional[bool] = False) -> None:
        super().__init__()
        self.is_jsonl = is_jsonl

    def load_data(self, input_file: str) -> List[Document]:
        documents = []
        with open(input_file, "r") as file:
            load_data = json.load(file)
        for data in load_data:
            metadata = {
                "title": data["title"],
                "published_at": data["published_at"],
                "source": data["source"],
            }
            documents.append(Document(text=data["body"], metadata=metadata))
        return documents


class CustomExtractor(BaseExtractor):
    async def aextract(self, nodes) -> List[Dict]:
        metadata_list = [
            {
                "title": (node.metadata["title"]),
                "source": (node.metadata["source"]),
                "published_at": (node.metadata["published_at"]),
            }
            for node in nodes
        ]
        return metadata_list


def gen_stage_0(corpus, queries, rank_model_name, rerank, rerank_model_name, output_name):
    openai.api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")
    openai.base_url = "your_api_base"
    voyage_api_key = os.environ.get("VOYAGE_API_KEY", "your_voyage_api_key")
    cohere_api_key = os.environ.get("COHERE_API_KEY", "your_cohere_api_key")
    model_name = rank_model_name
    def_llm = "gpt-3.5-turbo-1106"
    topk = 10
    chunk_size = 256
    context_window = 2048
    num_output = 256
    save_file = output_name
    model_name = rank_model_name
    llm = OpenAI(model=def_llm, temperature=0, max_tokens=context_window)

    print(f"Saving results to {save_file}. Previous data will be overwritten.")

    if "text" in model_name:
        embed_model = OpenAIEmbedding(model=model_name, embed_batch_size=10)
    elif "Cohere" in model_name:
        embed_model = CohereEmbedding(
            cohere_api_key=cohere_api_key, model_name="embed-english-v3.0", input_type="search_query"
        )
    elif "voyage-02" in model_name:
        embed_model = VoyageEmbedding(model_name="voyage-02", voyage_api_key=voyage_api_key)
    elif "instructor" in model_name:
        embed_model = InstructorEmbedding(model_name=model_name)
    else:
        embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)

    text_splitter = SentenceSplitter(chunk_size=chunk_size)

    prompt_helper = PromptHelper(
        context_window=context_window,
        num_output=num_output,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None,
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=text_splitter,
        prompt_helper=prompt_helper,
    )

    set_global_service_context(service_context)

    reader = JSONReader()
    data = reader.load_data(corpus)
    print("Corpus Data")
    print("--------------------------")
    print(data[0])
    print("--------------------------")

    print("Initialising pipeline")
    transformations = [text_splitter, CustomExtractor()]
    pipeline = IngestionPipeline(transformations=transformations)
    nodes = pipeline.run(documents=data)
    nodes_see = deepcopy(nodes)
    print("LLM sees:\n", (nodes_see)[0].get_content(metadata_mode=MetadataMode.LLM))
    print("Finished Loading...")

    index = VectorStoreIndex(nodes, show_progress=True)
    print("Vector Store Created ...")

    with open(queries, "r") as file:
        query_data = json.load(file)

    print("Query Data")
    print("--------------------------")
    print(query_data[0])
    print("--------------------------")

    if rerank:
        print("Reranker enabled")
        rerank_postprocessors = FlagEmbeddingReranker(model=rerank_model_name, top_n=topk)

    retrieval_save_list = []
    print("Running Retrieval ...")
    for data in tqdm(query_data):
        query = data["query"]
        if rerank:
            nodes_score = index.as_retriever(similarity_top_k=20).retrieve(query)
            nodes_score = rerank_postprocessors.postprocess_nodes(
                nodes_score, query_bundle=QueryBundle(query_str=query)
            )
        else:
            nodes_score = index.as_retriever(similarity_top_k=topk).retrieve(query)

        retrieval_list = []
        for ns in nodes_score:
            dic = {}
            dic["text"] = ns.get_content(metadata_mode=MetadataMode.LLM)
            dic["score"] = ns.get_score()
            retrieval_list.append(dic)

        save = {}
        save["query"] = data["query"]
        save["answer"] = data["answer"]
        save["question_type"] = data["question_type"]
        save["retrieval_list"] = retrieval_list
        save["gold_list"] = data["evidence_list"]
        retrieval_save_list.append(save)

    print("Retieval complete. Saving Results")
    with open(save_file, "w") as json_file:
        json.dump(retrieval_save_list, json_file)


def _peak_mem_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)


if __name__ == "__main__":
    if STAGING:
        corpus = "data/sample-corpus.json"
        queries = "data/sample-rag.json"
    else:
        corpus = "data/corpus.json"
        queries = "data/rag.json"

    rank_model_name = "BAAI/llm-embedder"
    output_name = "output/embedder-rankerA.json"
    rerank = True

    gen_stage_0(corpus, queries, rank_model_name, rerank, None, output_name)

    if STAGING:
        print(f"Peak working set size: {_peak_mem_gb():.2f} GB")
