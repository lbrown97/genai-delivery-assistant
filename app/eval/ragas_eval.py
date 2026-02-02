import json
import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from app.core.settings import settings
from app.llm.models import get_chat_model
from app.rag.embeddings import get_embedding_model
from app.rag.retriever import (
    clear_doc_scope_override,
    get_retriever_with_scores,
    set_doc_scope_override,
)

DATASET_PATH = Path("app/eval/datasets/questions.jsonl")
PDF_DATASET_PATH = Path("app/eval/datasets/pdf_questions.jsonl")
OUT_PATH = Path("app/eval/results.json")


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_all_questions() -> List[Dict[str, Any]]:
    rows = _load_questions(DATASET_PATH)
    rows.extend(_load_questions(PDF_DATASET_PATH))
    return rows


def _answer_from_context(question: str, contexts: List[str]) -> str:
    joined = "\n\n---\n\n".join(contexts)
    prompt = (
        "Answer the question using ONLY the provided context. "
        "If the context is insufficient, say: 'I don't know based on the provided documents.'\n\n"
        f"Context:\n{joined}\n\nQuestion:\n{question}"
    )
    llm = get_chat_model(temperature=0.0)
    msg = llm.invoke(prompt)
    return msg.content.strip()


def _build_ragas_llm():
    from openai import OpenAI

    base_url = os.getenv("OLLAMA_BASE_URL", settings.OLLAMA_BASE_URL).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    client = OpenAI(api_key="ollama", base_url=base_url)
    return llm_factory(settings.OLLAMA_MODEL, provider="openai", client=client)


def _build_ragas_embeddings():
    # Use the same LangChain embeddings as the app (GPU if available)
    return get_embedding_model()


def build_dataset(k: int = 3) -> Dataset:
    rows = _load_all_questions()
    include_external = bool(_load_questions(PDF_DATASET_PATH))
    if include_external:
        set_doc_scope_override("all")
    records: Dict[str, List[Any]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    try:
        for r in rows:
            question = r["question"]
            gt = r.get("ground_truth", "")
            pairs = get_retriever_with_scores(question, k=k)
            docs = [d for d, _ in pairs]
            contexts = [d.page_content for d in docs]
            answer = _answer_from_context(question, contexts)

            records["question"].append(question)
            records["answer"].append(answer)
            records["contexts"].append(contexts)
            records["ground_truth"].append(gt)
    finally:
        if include_external:
            clear_doc_scope_override()

    return Dataset.from_dict(records)


def run_eval():
    dataset = build_dataset()

    llm = _build_ragas_llm()
    embeddings = _build_ragas_embeddings()

    # Reduce generations for faster eval and fewer warnings with Ollama
    answer_relevancy.strictness = 1

    metrics = [
        answer_relevancy,
        context_precision,
        faithfulness,
        context_recall,
    ]

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        column_map={
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": "contexts",
            "reference": "ground_truth",
        },
        batch_size=8,
    )

    df = result.to_pandas()
    OUT_PATH.write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    run_eval()
