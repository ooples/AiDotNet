#!/usr/bin/env python3
"""LangGraph side of the AiDotNet head-to-head benchmark.

Builds a minimal LangGraph StateGraph with a single "model + tool" node driven by a deterministic in-process
mock (no LLM, no network) and times N graph invocations, mirroring the single tool-enabled turn the .NET
SemanticKernelComparisonBenchmark measures. Prints one JSON line so the C# runner can parse it.

Run standalone:  python langgraph_sidecar.py 2000
Requires:        pip install langgraph
"""
import json
import sys
import time

try:
    from typing import TypedDict
    from langgraph.graph import StateGraph, END
except Exception as exc:  # langgraph not installed / import failure
    print(json.dumps({"status": "unavailable", "error": str(exc)}))
    sys.exit(0)


class State(TypedDict):
    input: str
    output: str


def model_node(state: State) -> dict:
    # Mock "model decides + tool result" turn: a fixed answer, no network.
    return {"output": "a concise answer"}


def build_app():
    graph = StateGraph(State)
    graph.add_node("model", model_node)
    graph.set_entry_point("model")
    graph.add_edge("model", END)
    return graph.compile()


def main() -> None:
    try:
        iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    except ValueError:
        print(json.dumps({"status": "invalid_arguments", "error": "iterations must be an integer"}))
        sys.exit(2)

    if iterations <= 0:
        print(json.dumps({"status": "invalid_arguments", "error": "iterations must be > 0"}))
        sys.exit(2)

    app = build_app()

    request = {"input": "What is 2 + 3?", "output": ""}
    for _ in range(50):  # warmup
        app.invoke(request)

    start = time.perf_counter()
    for _ in range(iterations):
        app.invoke(request)
    elapsed = time.perf_counter() - start

    print(json.dumps({
        "status": "ok",
        "framework": "langgraph",
        "iterations": iterations,
        "mean_microseconds": round(elapsed / iterations * 1e6, 3),
    }))


if __name__ == "__main__":
    main()
