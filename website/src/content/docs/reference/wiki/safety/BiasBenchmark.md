---
title: "BiasBenchmark<T>"
description: "Benchmark for evaluating bias and fairness detection accuracy across demographic groups."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Benchmark for evaluating bias and fairness detection accuracy across demographic groups.

## For Beginners

This benchmark tests whether your AI can detect biased language —
things like stereotypes or unfair treatment of demographic groups. It uses examples of
biased and unbiased text and measures how accurately the system identifies bias.

## How It Works

Tests the safety pipeline against text containing stereotypes, demographic parity violations,
representational bias, and intersectional bias. Also includes balanced, fair text to measure
false positive rates on unbiased content.

**References:**

- BEATS: Comprehensive bias evaluation test suite for LLMs (2025, arxiv:2503.24310)
- SB-Bench: Stereotype bias benchmark for multimodal models (2025, arxiv:2502.08779)
- StereoSet: Measuring stereotypical bias (ACL 2021)
- CrowS-Pairs: Measuring social biases (EMNLP 2020)

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTestCases` |  |
| `RunBenchmark(SafetyPipeline<>)` |  |

