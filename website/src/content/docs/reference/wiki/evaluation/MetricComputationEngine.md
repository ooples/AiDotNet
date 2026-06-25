---
title: "MetricComputationEngine<T>"
description: "Core engine for computing evaluation metrics across all task types."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Core engine for computing evaluation metrics across all task types.

## For Beginners

This is the heart of model evaluation. Give it predictions and actuals,
and it computes all relevant metrics automatically based on task type.

## How It Works

This engine provides a unified interface for computing classification, regression,
and time series metrics with confidence intervals and proper handling of edge cases.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetricComputationEngine(EvaluationOptions<>)` | Initializes the metric computation engine with default or custom options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeClassificationMetrics(ReadOnlySpan<>,ReadOnlySpan<>,ReadOnlySpan<>,Int32)` | Computes all classification metrics. |
| `ComputeMetric(String,ReadOnlySpan<>,ReadOnlySpan<>,ReadOnlySpan<>,Int32)` | Computes a specific metric by name. |
| `ComputeRegressionMetrics(ReadOnlySpan<>,ReadOnlySpan<>)` | Computes all regression metrics. |
| `GetAvailableMetricNames(Boolean,Boolean,Boolean)` | Gets all available metric names. |
| `RegisterClassificationMetric(IClassificationMetric<>,String)` | Registers a classification metric. |
| `RegisterProbabilisticMetric(IProbabilisticClassificationMetric<>,String)` | Registers a probabilistic classification metric. |
| `RegisterRegressionMetric(IRegressionMetric<>,String)` | Registers a regression metric. |

