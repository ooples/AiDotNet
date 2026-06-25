---
title: "ISafetyBenchmark<T>"
description: "Interface for safety benchmark modules that evaluate safety module accuracy."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Benchmarking`

Interface for safety benchmark modules that evaluate safety module accuracy.

## For Beginners

A safety benchmark tests how well your safety modules work.
It feeds known examples (some safe, some harmful) through the modules and measures
how accurately they detect the harmful ones without blocking safe content.

## How It Works

Safety benchmarks run standardized test suites against safety modules to measure
detection accuracy, precision, recall, F1 scores, and false positive rates.
Each benchmark focuses on a specific safety domain (toxicity, jailbreak, PII, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` | Gets the name of this benchmark. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RunBenchmark(SafetyPipeline<>)` | Runs the benchmark against the given safety pipeline and returns results. |

