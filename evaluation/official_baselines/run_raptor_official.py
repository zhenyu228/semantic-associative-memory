from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from official_eval_utils import (
    answer_hit,
    load_common_inputs,
    rough_retrieved_doc_ids_from_text,
    summarize_answer_metrics,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 RAPTOR 官方实现运行 QA baseline")
    parser.add_argument("--prepared-dir", required=True, help="export_sam_for_official.py 生成的 prepared 目录")
    parser.add_argument("--external-dir", default="evaluation/external/raptor", help="RAPTOR 官方仓库目录")
    parser.add_argument("--output", default=None, help="结果 JSON 路径")
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少个问题")
    parser.add_argument("--client-type", default=None, choices=["openai", "azure"], help="API 客户端类型，默认读取 RAPTOR_CLIENT_TYPE")
    parser.add_argument("--azure-endpoint", default=None, help="Azure-style endpoint，默认读取 RAPTOR_AZURE_ENDPOINT 或 GPT54_BASE_URL")
    parser.add_argument("--api-version", default=None, help="Azure API version，默认读取 RAPTOR_API_VERSION 或 GPT54_API_VERSION")
    parser.add_argument("--qa-model", default=None, help="公司网关中的 chat/QA 模型名，默认读取 RAPTOR_QA_MODEL 或 gpt-3.5-turbo")
    parser.add_argument("--summary-model", default=None, help="公司网关中的摘要模型名，默认读取 RAPTOR_SUMMARY_MODEL 或 qa-model")
    parser.add_argument("--embedding-model", default=None, help="公司网关中的 embedding 模型名，默认读取 RAPTOR_EMBEDDING_MODEL 或 text-embedding-ada-002")
    parser.add_argument("--embedding-api-key", default=None, help="Azure embedding API key，默认读取 RAPTOR_EMBEDDING_API_KEY、EMBEDDING_API_KEY 或 SAM_AZURE_EMBEDDING_API_KEY")
    parser.add_argument("--embedding-azure-endpoint", default=None, help="Azure embedding endpoint，默认读取 RAPTOR_EMBEDDING_AZURE_ENDPOINT、EMBEDDING_BASE_URL 或 SAM_AZURE_EMBEDDING_ENDPOINT")
    parser.add_argument("--embedding-api-version", default=None, help="Azure embedding API version，默认读取 RAPTOR_EMBEDDING_API_VERSION、EMBEDDING_API_VERSION 或 SAM_AZURE_EMBEDDING_API_VERSION")
    parser.add_argument("--embedding-dimensions", type=int, default=None, help="可选：embedding 输出维度，例如 1024")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_dir = ROOT / args.external_dir
    if str(external_dir) not in sys.path:
        sys.path.insert(0, str(external_dir))
    try:
        from raptor import RetrievalAugmentation
        from raptor import RetrievalAugmentationConfig
        from raptor.EmbeddingModels import BaseEmbeddingModel
        from raptor.EmbeddingModels import OpenAIEmbeddingModel
        from raptor.QAModels import BaseQAModel
        from raptor.QAModels import GPT3TurboQAModel
        from raptor.SummarizationModels import BaseSummarizationModel
        from raptor.SummarizationModels import GPT3TurboSummarizationModel
    except Exception as exc:
        raise RuntimeError(
            "无法导入 RAPTOR 官方实现。请先运行 fetch_official_repos.py，"
            "并按 evaluation/official_baselines/README.md 安装官方依赖。"
        ) from exc

    prepared_dir = ROOT / args.prepared_dir
    documents, queries = load_common_inputs(prepared_dir)
    if args.limit:
        queries = queries[: args.limit]
    corpus_path = prepared_dir / "raptor/corpus.txt"
    corpus = corpus_path.read_text(encoding="utf-8")

    client_type = args.client_type or _env("RAPTOR_CLIENT_TYPE", "openai")
    qa_model_name = args.qa_model or _env("RAPTOR_QA_MODEL", "gpt-3.5-turbo")
    summary_model_name = args.summary_model or _env("RAPTOR_SUMMARY_MODEL", qa_model_name)
    embedding_model_name = (
        args.embedding_model
        or _env("RAPTOR_EMBEDDING_MODEL", None)
        or _env("EMBEDDING_MODEL", None)
        or _env("SAM_AZURE_EMBEDDING_MODEL", "text-embedding-ada-002")
    )
    api_key = _env("OPENAI_API_KEY", None) or _env("GPT54_API_KEY", None)

    if client_type == "azure":
        azure_endpoint = args.azure_endpoint or _env("RAPTOR_AZURE_ENDPOINT", None) or _env("GPT54_BASE_URL", None)
        api_version = args.api_version or _env("RAPTOR_API_VERSION", None) or _env("GPT54_API_VERSION", "2024-02-01")
        if not api_key:
            raise RuntimeError("RAPTOR Azure 模式需要 OPENAI_API_KEY 或 GPT54_API_KEY。")
        if not azure_endpoint:
            raise RuntimeError("RAPTOR Azure 模式需要 RAPTOR_AZURE_ENDPOINT 或 GPT54_BASE_URL。")
        embedding_api_key = (
            args.embedding_api_key
            or _env("RAPTOR_EMBEDDING_API_KEY", None)
            or _env("EMBEDDING_API_KEY", None)
            or _env("SAM_AZURE_EMBEDDING_API_KEY", None)
            or api_key
        )
        embedding_azure_endpoint = (
            args.embedding_azure_endpoint
            or _env("RAPTOR_EMBEDDING_AZURE_ENDPOINT", None)
            or _env("EMBEDDING_BASE_URL", None)
            or _env("SAM_AZURE_EMBEDDING_ENDPOINT", None)
            or azure_endpoint
        )
        embedding_api_version = (
            args.embedding_api_version
            or _env("RAPTOR_EMBEDDING_API_VERSION", None)
            or _env("EMBEDDING_API_VERSION", None)
            or _env("SAM_AZURE_EMBEDDING_API_VERSION", None)
            or api_version
        )
        embedding_dimensions = (
            args.embedding_dimensions
            or _int_env("RAPTOR_EMBEDDING_DIMENSIONS")
            or _int_env("EMBEDDING_DIMENSIONS")
            or _int_env("SAM_AZURE_EMBEDDING_DIMENSIONS")
        )
        qa_model = AzureChatQAModel(
            BaseQAModel=BaseQAModel,
            model=qa_model_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        summarization_model = AzureChatSummarizationModel(
            BaseSummarizationModel=BaseSummarizationModel,
            model=summary_model_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        embedding_model = AzureEmbeddingModel(
            BaseEmbeddingModel=BaseEmbeddingModel,
            model=embedding_model_name,
            api_key=embedding_api_key,
            azure_endpoint=embedding_azure_endpoint,
            api_version=embedding_api_version,
            dimensions=embedding_dimensions,
        )
    else:
        qa_model = GPT3TurboQAModel(model=qa_model_name)
        summarization_model = GPT3TurboSummarizationModel(model=summary_model_name)
        embedding_model = OpenAIEmbeddingModel(model=embedding_model_name)

    retrieval_augmentation = RetrievalAugmentation(
        config=RetrievalAugmentationConfig(
            qa_model=qa_model,
            summarization_model=summarization_model,
            embedding_model=embedding_model,
        )
    )
    retrieval_augmentation.add_documents(corpus)

    results = []
    for query in queries:
        answer = retrieval_augmentation.answer_question(question=query["question"])
        retrieved_doc_ids = rough_retrieved_doc_ids_from_text(answer, documents)
        results.append(
            {
                "query_id": query["id"],
                "question": query["question"],
                "gold_answers": query["answers"],
                "official_answer": answer,
                "answer_hit": answer_hit(answer, query["answers"]),
                "retrieved_doc_ids_diagnostic": retrieved_doc_ids,
                "evidence_recall": None,
                "note": "RAPTOR 官方高层 QA API 返回答案文本；doc id 为文本反查诊断，不作为严格官方检索输出。",
            }
        )

    output = Path(args.output) if args.output else prepared_dir.parent / "results/raptor_official.json"
    write_json(
        output,
        {
            "method": "raptor_official",
            "official_repo": "https://github.com/parthsarthi03/raptor",
            "models": {
                "client_type": client_type,
                "qa_model": qa_model_name,
                "summary_model": summary_model_name,
                "embedding_model": embedding_model_name,
            },
            "metrics": summarize_answer_metrics(results),
            "results": results,
        },
    )
    print(f"RAPTOR 官方评测结果：{output}")


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


def _azure_client(api_key: str, azure_endpoint: str, api_version: str):
    return {
        "api_key": api_key,
        "azure_endpoint": azure_endpoint,
        "api_version": api_version,
    }


def AzureChatQAModel(BaseQAModel, model: str, api_key: str, azure_endpoint: str, api_version: str):
    class _AzureChatQAModel(BaseQAModel):
        def __init__(self) -> None:
            self.model = model
            self.client = _azure_client(api_key, azure_endpoint, api_version)

        def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
            response = _chat_completion(
                client=self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a question answering assistant."},
                    {
                        "role": "user",
                        "content": f"Given Context: {context}\nAnswer the question as directly as possible: {question}",
                    },
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

    return _AzureChatQAModel()


def AzureChatSummarizationModel(
    BaseSummarizationModel,
    model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
):
    class _AzureChatSummarizationModel(BaseSummarizationModel):
        def __init__(self) -> None:
            self.model = model
            self.client = _azure_client(api_key, azure_endpoint, api_version)

        def summarize(self, context, max_tokens=500, stop_sequence=None):
            response = _chat_completion(
                client=self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

    return _AzureChatSummarizationModel()


def AzureEmbeddingModel(
    BaseEmbeddingModel,
    model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    dimensions: int | None = None,
):
    class _AzureEmbeddingModel(BaseEmbeddingModel):
        def __init__(self) -> None:
            self.model = model
            self.client = _azure_client(api_key, azure_endpoint, api_version)

        def create_embedding(self, text):
            text = text.replace("\n", " ")
            return _embedding(self.client, self.model, text, dimensions=dimensions)

    return _AzureEmbeddingModel()


def _chat_completion(client, model: str, messages: list[dict], max_tokens: int):
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": max_tokens,
    }
    response = _post_azure(
        client=client,
        deployment=model,
        endpoint="chat/completions",
        payload=payload,
    )
    return _DictCompletion(response)


def _embedding(client, model: str, text: str, dimensions: int | None = None) -> list[float]:
    payload: dict[str, object] = {"input": [text], "model": model}
    if dimensions:
        payload["dimensions"] = dimensions
    response = _post_azure(
        client=client,
        deployment=model,
        endpoint="embeddings",
        payload=payload,
    )
    return response["data"][0]["embedding"]


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _post_azure(client: dict, deployment: str, endpoint: str, payload: dict) -> dict:
    url = (
        f"{client['azure_endpoint'].rstrip('/')}/openai/deployments/{deployment}/"
        f"{endpoint}?api-version={client['api_version']}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": client["api_key"]},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Azure-style API 请求失败：HTTP {exc.code} {body[:500]}") from exc


class _DictCompletion:
    def __init__(self, response: dict) -> None:
        self.choices = [
            _DictChoice(choice.get("message", {}).get("content", ""))
            for choice in response.get("choices", [])
        ]


class _DictChoice:
    def __init__(self, content: str) -> None:
        self.message = _DictMessage(content)


class _DictMessage:
    def __init__(self, content: str) -> None:
        self.content = content


if __name__ == "__main__":
    main()
