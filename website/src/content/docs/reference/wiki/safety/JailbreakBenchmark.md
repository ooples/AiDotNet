---
title: "JailbreakBenchmark<T>"
description: "Benchmark for evaluating jailbreak and prompt injection detection accuracy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Benchmark for evaluating jailbreak and prompt injection detection accuracy.

## For Beginners

A "jailbreak" is when someone tricks an AI into ignoring its safety
rules. This benchmark tests how well your safety pipeline detects these tricks. It includes
both known attack patterns and normal questions to make sure the system doesn't block safe content.

## How It Works

Tests the safety pipeline against known jailbreak attack patterns including direct overrides,
persona attacks (DAN), role-playing attacks, encoding-based attacks, multi-turn escalation,
and prompt extraction. Also includes benign prompts to measure false positive rates.

**References:**

- JailbreakBench: Open robustness benchmark for jailbreaking (NeurIPS 2024)
- SORRY-Bench: 450 linguistically diverse unsafe requests (ICLR 2025)
- HarmBench: Standardized evaluation of automated red teaming (ICML 2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTestCases` |  |
| `RunBenchmark(SafetyPipeline<>)` |  |

