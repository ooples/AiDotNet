---
title: "SafetyBenchmarkRunner<T>"
description: "Runs safety benchmarks against a configured safety pipeline to measure detection performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Runs safety benchmarks against a configured safety pipeline to measure detection performance.

## For Beginners

Before deploying a safety system, you need to test how well it works.
This benchmark runner tests your safety pipeline against known examples of safe and unsafe
content and tells you how accurate it is. Higher precision means fewer false alarms;
higher recall means fewer missed violations.

## How It Works

Evaluates a safety pipeline against standard benchmark test cases to measure precision,
recall, F1 score, and false positive rate across safety categories. Results can be used
to tune thresholds and compare different pipeline configurations.

**Metrics computed:**

- Precision: Of flagged content, how much was actually unsafe?
- Recall: Of all unsafe content, how much was correctly flagged?
- F1 Score: Harmonic mean of precision and recall
- False Positive Rate: Percentage of safe content incorrectly flagged
- Latency: Average evaluation time per test case

**References:**

- SafetyBench: Multi-category safety evaluation (ACL 2024)
- HarmBench: Standardized evaluation of automated red teaming (ICML 2024)
- SimpleSafetyTests: 100 test prompts for critical safety risks (Vidgen et al., 2024)
- WildGuardTest: 1.7k labeled examples across 13 risk categories (Allen AI, 2024)
- SORRY-Bench: 450 linguistically diverse unsafe requests (ICLR 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SafetyBenchmarkRunner(SafetyPipeline<>)` | Initializes a new safety benchmark runner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RunBenchmark(IReadOnlyList<SafetyBenchmarkCase>)` | Runs the benchmark suite and returns a summary report. |

