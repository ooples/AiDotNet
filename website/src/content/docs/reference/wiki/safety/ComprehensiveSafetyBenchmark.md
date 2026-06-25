---
title: "ComprehensiveSafetyBenchmark<T>"
description: "Comprehensive safety benchmark that aggregates all individual benchmarks into a single suite."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Comprehensive safety benchmark that aggregates all individual benchmarks into a single suite.

## For Beginners

Instead of running each safety benchmark separately, this runs
all of them at once and gives you a complete picture of how well your safety system works
across all categories. Use this for a one-stop safety evaluation.

## How It Works

Runs all available safety benchmarks (toxicity, jailbreak, bias, PII, hallucination, watermark,
adversarial) and produces a unified report with per-category breakdowns. This is the recommended
benchmark to run for a complete safety evaluation of your pipeline.

**References:**

- SafetyBench: Multi-category safety evaluation (ACL 2024)
- HarmBench: Standardized evaluation across attack types (ICML 2024)
- WildGuardTest: 1.7k labeled examples across 13 risk categories (Allen AI, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComprehensiveSafetyBenchmark` | Initializes a new comprehensive safety benchmark with all default sub-benchmarks. |
| `ComprehensiveSafetyBenchmark(IReadOnlyList<ISafetyBenchmark<>>)` | Initializes a new comprehensive safety benchmark with specific sub-benchmarks. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTestCases` |  |
| `RunAllBenchmarks(SafetyPipeline<>)` | Runs all sub-benchmarks and returns individual results keyed by benchmark name. |
| `RunBenchmark(SafetyPipeline<>)` |  |

