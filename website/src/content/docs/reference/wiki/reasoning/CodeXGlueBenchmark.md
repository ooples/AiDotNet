---
title: "CodeXGlueBenchmark<T>"
description: "CodeXGLUE benchmark harness (dataset-loader + metric computation)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

CodeXGLUE benchmark harness (dataset-loader + metric computation).

## How It Works

CodeXGLUE is a suite of code understanding and code generation tasks. This harness is intentionally "dataset
agnostic": callers provide a JSONL file path and field mapping, and provide the model invocation function.

This harness does not attempt to ship, download, or cache CodeXGLUE datasets; it only provides the evaluation glue.

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |
| `Description` |  |
| `TotalProblems` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAsync(Func<String,Task<String>>,Nullable<Int32>,CancellationToken)` |  |
| `LoadProblemsAsync(Nullable<Int32>)` |  |

