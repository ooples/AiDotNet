---
title: "HumanEvalDataLoader"
description: "Loader for HumanEval benchmark dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks.Data`

Loader for HumanEval benchmark dataset.

## How It Works

HumanEval format:
{
"task_id": "HumanEval/0",
"prompt": "def has_close_elements(numbers, threshold):\n \"\"\" Check if in given list...",
"canonical_solution": " for idx, elem in enumerate(numbers):...",
"test": "def check(candidate):...",
"entry_point": "has_close_elements"
}

