---
title: "PIIBenchmark<T>"
description: "Benchmark for evaluating PII (Personally Identifiable Information) detection accuracy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Benchmark for evaluating PII (Personally Identifiable Information) detection accuracy.

## For Beginners

PII includes things like email addresses, phone numbers, and Social
Security numbers. This benchmark tests whether your AI can detect when someone's private
information is being exposed, while not false-flagging regular numbers or addresses.

## How It Works

Tests the safety pipeline against text containing various types of PII: email addresses,
Social Security numbers, credit card numbers, phone numbers, IP addresses, and names.
Also includes text with number-like patterns that are NOT PII to measure false positive rates.

**References:**

- Microsoft Presidio: PII detection and anonymization (2024)
- AI Village CTF: PII leakage challenge (DEF CON 31, 2023)

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTestCases` |  |
| `RunBenchmark(SafetyPipeline<>)` |  |

