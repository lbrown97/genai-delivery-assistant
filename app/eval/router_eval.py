import json
from pathlib import Path

from app.agent.router import select_tool_call

DATASET_PATH = Path("app/eval/datasets/router_eval.jsonl")
OUT_PATH = Path("app/eval/router_eval_results.json")


def _load_rows():
    """Load router-evaluation rows from JSONL dataset."""

    rows = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def run_eval():
    """Run router tool-selection evaluation and write result summary JSON."""

    rows = _load_rows()

    results = []
    correct = 0
    for r in rows:
        question = r["question"]
        expected = r["expected_tool"]
        try:
            call = select_tool_call(question)
            predicted = call.tool
        except Exception:
            predicted = "invalid"

        ok = predicted == expected
        if ok:
            correct += 1
        results.append(
            {
                "question": question,
                "expected_tool": expected,
                "predicted_tool": predicted,
                "correct": ok,
            }
        )

    summary = {
        "total": len(rows),
        "correct": correct,
        "accuracy": (correct / len(rows)) if rows else 0.0,
    }

    OUT_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(run_eval())
