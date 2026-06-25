---
title: "StandardSafetyBenchmarks"
description: "Provides standard safety benchmark test suites inspired by published safety evaluation datasets."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Safety.Benchmarking`

Provides standard safety benchmark test suites inspired by published safety evaluation datasets.

## For Beginners

These are ready-to-use test sets you can run against your safety
pipeline. Each set contains examples of both safe and unsafe content for a specific category.
Use these to verify your safety system is working before deploying to production.

## How It Works

Contains curated test cases for evaluating safety pipeline effectiveness across different
categories. These are simplified versions of research benchmark datasets, designed to
provide a quick smoke test of safety capabilities.

**Inspired by:**

- SimpleSafetyTests: 100 test prompts across 5 critical safety risks (Vidgen et al., 2024)
- HarmBench: Standardized evaluation of automated red teaming (ICML 2024)
- SORRY-Bench: 450 linguistically diverse unsafe requests (ICLR 2025)
- WildGuardTest: 1.7k labeled examples across 13 risk categories (Allen AI, 2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `FullBenchmark` | Gets the full standard benchmark combining all categories. |
| `JailbreakBenchmark` | Gets a basic jailbreak detection benchmark with known attack patterns and benign prompts. |
| `PIIBenchmark` | Gets a PII detection benchmark with known PII patterns and clean text. |
| `ToxicityBenchmark` | Gets a toxicity detection benchmark with toxic and non-toxic content. |

