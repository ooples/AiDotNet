---
title: "AdversarialBenchmark<T>"
description: "Benchmark for evaluating adversarial robustness of the safety pipeline against evasion attacks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Benchmark for evaluating adversarial robustness of the safety pipeline against evasion attacks.

## For Beginners

Attackers try to get around safety filters by subtly changing their
text — like using numbers instead of letters ("h4te") or encoding messages. This benchmark
tests whether your safety system can still catch harmful content even when it's been disguised.

## How It Works

Tests the safety pipeline against adversarial perturbations designed to evade detection:
character substitution (l33tspeak, homoglyphs), word reordering, synonym replacement to evade
keyword lists, encoding attacks (base64, rot13 references), and multi-language evasion.
Measures how well the system maintains detection accuracy under adversarial conditions.

**References:**

- AutoDAN: Automated red teaming for LLMs (ICLR 2024)
- GCG: Universal and transferable adversarial attacks (NeurIPS 2023)
- TextBugger: Adversarial text generation (NDSS 2019)

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTestCases` |  |
| `RunBenchmark(SafetyPipeline<>)` |  |

