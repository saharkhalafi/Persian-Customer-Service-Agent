import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.core.container import build_agent
from app.core.context import RequestContext
from app.core.database import SessionLocal
from app.core.gemini import GeminiClient

OUT = Path(__file__).resolve().parent / "results" / "llm_latency_sample.json"
CASES = [
    ("greeting", "سلام"),
    ("order", "جمعاً چقدر خرید کردم؟"),
    ("product", "کرم ضد چروک پرایم دارید؟"),
]


def main() -> None:
    db = SessionLocal()
    rows = []
    try:
        agent = build_agent(db=db, gemini=GeminiClient())
        for name, question in CASES:
            started = time.perf_counter()
            result = agent.run(
                user_message=question,
                context=RequestContext(customer_id="9206288"),
            )
            rows.append(
                {
                    "case": name,
                    "model": agent.gemini.model,
                    "thinking_budget": agent.gemini.thinking_budget,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "tools": [call.name for call in result.tool_calls],
                }
            )
            print(rows[-1])
    finally:
        db.close()
    OUT.write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
