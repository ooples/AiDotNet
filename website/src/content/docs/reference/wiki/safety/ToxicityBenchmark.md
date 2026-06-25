---
title: "ToxicityBenchmark<T>"
description: "Benchmark for evaluating toxicity detection accuracy across hate speech, threats, and harassment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Benchmark for evaluating toxicity detection accuracy across hate speech, threats, and harassment.

## For Beginners

This benchmark tests how well your AI can detect toxic language —
things like threats, slurs, and harassment. It uses known examples of toxic and safe text
and tells you how many the system caught and how many it missed.

## How It Works

Evaluates the safety pipeline against a curated set of toxic and non-toxic content samples.
Measures precision, recall, and F1 for detecting violence threats, hate speech, harassment,
and other harmful language patterns.

**References:**

- Jigsaw Toxic Comment Classification (Kaggle, 2018)
- HateXplain: Benchmark for explainable hate speech detection (AAAI 2021)
- ToxiGen: Machine-generated implicit hate speech dataset (ACL 2022)

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTestCases` |  |
| `RunBenchmark(SafetyPipeline<>)` |  |

