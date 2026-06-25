---
title: "MetricWithCI<T>"
description: "Represents a metric value with optional confidence interval and metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Results.Core`

Represents a metric value with optional confidence interval and metadata.

## For Beginners

When you measure something, you get a single number (the point estimate).
But that number has uncertainty. The confidence interval tells you the range where the true
value likely falls. For example: "Accuracy = 0.85 [0.82, 0.88]" means the accuracy is about
85%, but could reasonably be anywhere from 82% to 88%.

## How It Works

This is the fundamental building block for all evaluation metrics. It contains the point
estimate, confidence interval, and metadata about how the metric was computed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetricWithCI` | Default parameterless constructor for serialization. |
| `MetricWithCI(,,,Double,ConfidenceIntervalMethod,String,MetricDirection)` | Initializes a new metric with a confidence interval. |
| `MetricWithCI(,String,MetricDirection)` | Initializes a new metric with just a point estimate. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CIMethod` | Method used to compute the confidence interval. |
| `Category` | Category of the metric (e.g., "Classification", "Regression"). |
| `ConfidenceLevel` | Confidence level (e.g., 0.95 for 95% CI). |
| `Description` | Human-readable description of the metric. |
| `Direction` | Whether higher values indicate better performance for this metric. |
| `HasConfidenceInterval` | Whether a confidence interval is available. |
| `IntervalWidth` | Width of the confidence interval (upper - lower). |
| `IsValid` | Whether this metric value is valid (not NaN or infinite). |
| `LowerBound` | Lower bound of the confidence interval. |
| `MarginOfError` | Half-width of the confidence interval (margin of error). |
| `Name` | Name of the metric (e.g., "Accuracy", "MSE"). |
| `SampleCount` | Number of samples used to compute the metric. |
| `StandardDeviation` | Standard deviation of the metric (e.g., across CV folds). |
| `StandardError` | Standard error of the metric. |
| `UpperBound` | Upper bound of the confidence interval. |
| `Value` | The point estimate (main value) of the metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Format(Int32,Boolean)` | Formats the metric for display. |
| `FormatAsPercentage(Int32,Boolean)` | Formats the metric as a percentage. |
| `IsSignificantlyBetterThan(MetricWithCI<>)` | Checks if this metric is significantly better than another at the given significance level. |
| `ToString` | Returns a string representation of the metric. |

