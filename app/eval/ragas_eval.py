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

from app.agent.guardrails import groundedness_with_scores
from app.agent.router_utils import (
    UNKNOWN_ANSWER,
    generate_ask_draft,
    groundedness_min_score,
    normalize_ask_answer,
)
from app.agent.tools import retrieve_context_with_scores
from app.core.env import env_float
from app.core.settings import settings
from app.rag.embeddings import get_embedding_model
from app.rag.retriever import (
    clear_doc_scope_override,
    set_doc_scope_override,
)

DATASET_PATH = Path("app/eval/datasets/questions.jsonl")
PDF_DATASET_PATH = Path("app/eval/datasets/pdf_questions.jsonl")
OUT_PATH = Path("app/eval/results.json")
SUMMARY_OUT_PATH = Path("app/eval/results.summary.json")


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load evaluation questions from a JSONL file."""

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
    """Load base and PDF-focused evaluation question sets."""

    rows = _load_questions(DATASET_PATH)
    rows.extend(_load_questions(PDF_DATASET_PATH))
    return rows


def _build_ragas_llm():
    """Create a RAGAS-compatible LLM wrapper pointed at local Ollama."""

    from openai import OpenAI

    base_url = os.getenv("OLLAMA_BASE_URL", settings.OLLAMA_BASE_URL).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    client = OpenAI(api_key="ollama", base_url=base_url)
    return llm_factory(settings.OLLAMA_MODEL, provider="openai", client=client)


def _build_ragas_embeddings():
    """Use the same embedding model configuration as the app runtime."""

    # Use the same LangChain embeddings as the app (GPU if available)
    return get_embedding_model()


def build_dataset(k: int = 3) -> Dataset:
    """Build a RAGAS dataset by retrieving context and generating answers."""

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
            retrieved = retrieve_context_with_scores(question, k=k)
            docs = retrieved["docs"]
            contexts = [d.page_content for d in docs]
            if not groundedness_with_scores(
                docs,
                retrieved.get("scores"),
                min_docs=1,
                min_unique_sources=1,
                min_score=groundedness_min_score(0.8),
            ):
                answer = UNKNOWN_ANSWER
            else:
                draft = generate_ask_draft(
                    question,
                    context=retrieved["context"],
                    sources=retrieved["sources"],
                    temperature=env_float("ASK_TEMPERATURE", 0.1),
                )
                answer, _ = normalize_ask_answer(draft, retrieved["sources"])

            records["question"].append(question)
            records["answer"].append(answer)
            records["contexts"].append(contexts)
            records["ground_truth"].append(gt)
    finally:
        if include_external:
            clear_doc_scope_override()

    return Dataset.from_dict(records)


def run_eval():
    """Run RAGAS evaluation and persist results to JSON."""

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
    rows = df.to_dict(orient="records")
    metric_means = {}
    for col in df.columns:
        series = df[col]
        if getattr(series, "dtype", None).kind not in {"i", "u", "f"}:
            continue
        value = series.dropna().mean()
        if value == value:  # NaN check without importing math
            metric_means[col] = round(float(value), 4)

    summary = {
        "rows": len(rows),
        "metrics": metric_means,
    }

    OUT_PATH.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    SUMMARY_OUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run_eval()
