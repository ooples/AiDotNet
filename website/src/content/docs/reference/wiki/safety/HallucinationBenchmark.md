---
title: "HallucinationBenchmark<T>"
description: "Benchmark for evaluating hallucination and factual grounding detection accuracy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Benchmark for evaluating hallucination and factual grounding detection accuracy.

## For Beginners

"Hallucination" is when an AI makes something up that sounds real
but isn't — like inventing a fake research paper or citing false statistics. This benchmark
tests whether your safety system can detect these fabrications.

## How It Works

Tests the safety pipeline against text containing factual claims, fabricated citations,
invented statistics, and self-contradictions. Also includes factually grounded text to
measure false positive rates.

**References:**

- HaluEval: Hallucination evaluation for LLMs (ACL 2023)
- FActScore: Fine-grained atomic evaluation of factual precision (EMNLP 2023)
- FELM: Benchmarking factuality evaluation of LLMs (NeurIPS 2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTestCases` |  |
| `RunBenchmark(SafetyPipeline<>)` |  |

