"""Small live A/B: current flash (thinking 0) vs flash-lite.

Does not change production ranking. Writes evaluation/results/llm_latency_ab.json.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from app.core.context import RequestContext
from app.core.database import SessionLocal
from app.core.gemini import GeminiClient
from app.core.container import build_agent

OUT = PROJECT_ROOT / "evaluation" / "results" / "llm_latency_ab.json"
CASES = [
    ("greeting", "سلام"),
    ("order", "جمعاً چقدر خرید کردم؟"),
    ("product", "کرم ضد چروک پرایم دارید؟"),
]


def run_case(model: str, question: str, customer_id: str) -> dict:
    os.environ["AGENT_MODEL"] = model
    os.environ["AGENT_THINKING_BUDGET"] = "0"
    db = SessionLocal()
    try:
        agent = build_agent(db=db, gemini=GeminiClient())
        started = time.perf_counter()
        result = agent.run(
            user_message=question,
            context=RequestContext(customer_id=customer_id),
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "model": model,
            "question": question,
            "latency_ms": latency_ms,
            "tools": [call.name for call in result.tool_calls],
            "answer_len": len(result.answer or ""),
        }
    finally:
        db.close()


def main() -> None:
    customer_id = os.getenv("EVAL_CUSTOMER_ID", "9206288")
    models = [
        os.getenv("AGENT_MODEL", "gemini-2.5-flash"),
        "gemini-2.5-flash-lite",
    ]
    rows = []
    for model in models:
        for name, question in CASES:
            print(f"{model} / {name}")
            try:
                row = run_case(model, question, customer_id)
                row["case"] = name
                rows.append(row)
                print(row)
            except Exception as exc:
                rows.append(
                    {
                        "model": model,
                        "case": name,
                        "question": question,
                        "error": type(exc).__name__,
                    }
                )
    payload = {"results": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
