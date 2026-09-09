import json
import time
from pathlib import Path
from typing import Any

from app.core.container import build_agent
from app.core.database import SessionLocal
from app.core.gemini import GeminiClient
from app.core.context import RequestContext


class EvaluationRunner:
    def __init__(self, dataset_path: str, customer_id: str):
        self.dataset_path = Path(dataset_path)
        self.customer_id = customer_id

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.db = SessionLocal()
        self.gemini = GeminiClient()
        self.agent = build_agent(
            db=self.db,
            gemini=self.gemini,
        )

        self.context = RequestContext(
            customer_id=str(customer_id)
        )

    def close(self):
        self.db.close()

    def run_case(
        self,
        case: dict[str, Any],
        suite: str,
    ) -> dict[str, Any]:

        query = case["query"]

        started = time.perf_counter()

        try:
            result = self.agent.run(
                user_message=query,
                context=self.context,
            )

            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

            actual_tools = [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in result.tool_calls
            ]

            return {
                "id": case["id"],
                "suite": suite,
                "query": query,
                "difficulty": case.get("difficulty"),
                "expected_intent": case.get("expected_intent"),
                "expected_route": case.get("expected_route"),
                "expected_tool": case.get("expected_tool"),
                "expected_arguments": case.get("expected_arguments"),
                "expected_behavior": case.get("expected_behavior"),
                "actual_answer": result.answer,
                "actual_tools": actual_tools,
                "latency_ms": latency_ms,
                "error": None,
            }

        except Exception as e:
            return {
                "id": case["id"],
                "suite": suite,
                "query": query,
                "difficulty": case.get("difficulty"),
                "expected_intent": case.get("expected_intent"),
                "expected_route": case.get("expected_route"),
                "expected_tool": case.get("expected_tool"),
                "expected_arguments": case.get("expected_arguments"),
                "expected_behavior": case.get("expected_behavior"),
                "actual_answer": "",
                "actual_tools": [],
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "error": str(e),
            }

    def run(self) -> list[dict[str, Any]]:
        results = []

        suites = [
            ("intent_router", self.dataset.get("intent_router", [])),
            ("tool_calling", self.dataset.get("tool_calling", [])),
            ("guardrails", self.dataset.get("guardrails", [])),
        ]

        try:
            for suite_name, cases in suites:
                for case in cases:
                    print(
                        f"[{suite_name}] "
                        f"{case['id']} "
                        f"{case['query']}"
                    )

                    result = self.run_case(
                        case=case,
                        suite=suite_name,
                    )

                    results.append(result)

        finally:
            self.close()

        return results