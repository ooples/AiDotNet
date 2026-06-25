---
title: "VerticalFederatedBenchmark<T>"
description: "Provides benchmarking utilities for evaluating VFL implementations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Provides benchmarking utilities for evaluating VFL implementations.

## For Beginners

When developing VFL systems, it's important to test with controlled
datasets to verify that the system works correctly and to measure performance. This class
generates synthetic benchmark datasets and runs standardized evaluation suites.

## How It Works

**Benchmark scenarios:**

**Reference:** VertiBench (ICLR 2024): "Feature distribution diversity matters for
fair and accurate VFL evaluation."

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateDataset(Int32,Int32,Int32,Double,Nullable<Int32>)` | Generates a synthetic vertically-partitioned dataset for benchmarking. |
| `RunBenchmark(VerticalFederatedLearningOptions,VflBenchmarkDataset<>)` | Runs a standardized benchmark suite on a VFL implementation. |

