---
title: "SafetyBenchmarkBase<T>"
description: "Abstract base class for safety benchmark modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Benchmarking`

Abstract base class for safety benchmark modules.

## For Beginners

This base class provides common code for all safety benchmarks.
Each benchmark type extends this and adds its own set of test cases to evaluate
how well your safety modules detect specific types of harmful content.

## How It Works

Provides shared infrastructure for safety benchmarks including test case execution,
metric calculation, and result aggregation. Concrete implementations provide
the specific test cases for each safety domain (toxicity, jailbreak, PII, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTestCases` | Gets the test cases for this benchmark. |
| `RunBenchmark(SafetyPipeline<>)` |  |
| `RunTestCases(SafetyPipeline<>,IReadOnlyList<SafetyBenchmarkCase>)` | Runs all test cases against the pipeline and computes metrics. |

