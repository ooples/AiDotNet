---
title: "WatermarkBenchmark<T>"
description: "Benchmark for evaluating watermark detection accuracy across text watermarking techniques."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Benchmark for evaluating watermark detection accuracy across text watermarking techniques.

## For Beginners

Watermarks are hidden signatures in AI-generated text that help
identify it as AI-generated. This benchmark tests whether your system can detect various
types of watermarks without false-flagging normal text.

## How It Works

Tests the safety pipeline against text that exhibits watermarking patterns (green list bias,
synonym substitution, syntactic manipulation) and text that is naturally written without
watermarks. Measures false positive rates to ensure the system doesn't flag all AI-generated
text as watermarked.

**References:**

- SynthID-Text: Watermarking for LLMs (Google DeepMind, 2024)
- A Watermark for Large Language Models (Kirchenbauer et al., ICML 2023)
- Unforgeable watermarking for language models (Christ et al., 2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTestCases` |  |
| `RunBenchmark(SafetyPipeline<>)` |  |

