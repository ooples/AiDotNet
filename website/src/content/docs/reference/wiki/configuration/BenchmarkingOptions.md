---
title: "BenchmarkingOptions"
description: "Configuration options for running benchmarks through the AiDotNet facade."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for running benchmarks through the AiDotNet facade.

## For Beginners

Benchmarking is how you measure model quality using standardized tests.
You choose a suite (like GSM8K for math reasoning), and AiDotNet runs it and returns a report.

## How It Works

This options class is designed for use with `AiModelBuilder.ConfigureBenchmarking(...)` and
`AiModelResult.EvaluateBenchmarksAsync(...)`. Users specify which benchmark suites to run,
and AiDotNet orchestrates the execution behind the scenes.

## Properties

| Property | Summary |
|:-----|:--------|
| `AttachReportToResult` | Gets or sets whether the generated report should be attached to the model result. |
| `CiMode` | Gets or sets whether benchmarking should run in CI-friendly mode. |
| `DetailLevel` | Gets or sets how much detail should be included in reports. |
| `FailurePolicy` | Gets or sets how failures should be handled. |
| `Leaf` | Gets or sets LEAF federated benchmark configuration (required when running `LEAF`). |
| `SampleSize` | Gets or sets an optional sample size for suites that support sampling. |
| `Seed` | Gets or sets an optional deterministic seed used for CI-friendly sampling. |
| `Suites` | Gets or sets the benchmark suites to run. |
| `Tabular` | Gets or sets federated tabular benchmark configuration (synthetic non-IID suite). |
| `Text` | Gets or sets federated text benchmark configuration (Sent140/Shakespeare suites). |
| `Vision` | Gets or sets federated vision benchmark configuration (FEMNIST/CIFAR suites). |

