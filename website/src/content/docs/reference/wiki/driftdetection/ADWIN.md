---
title: "ADWIN<T>"
description: "ADWIN<T> — Models & Types in AiDotNet.DriftDetection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DriftDetection`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ADWIN(Double,Int32)` | Creates a new ADWIN drift detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Delta` | Gets the delta parameter (confidence level for drift detection). |
| `WindowSize` | Gets the current window size (number of observations in memory). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObservation()` | Adds a new observation and checks for drift. |
| `CalculateEpsilon(Int32,Int32)` | Calculates the epsilon threshold using the Hoeffding bound. |
| `CheckDrift` | Checks if drift has occurred by finding a significant cut point. |
| `CompressBuckets` | Compresses buckets when there are too many at a given level. |
| `EstimateDriftProbability` | Estimates the probability of drift based on current window statistics. |
| `InsertElement(Double)` | Inserts a new element into the bucket structure. |
| `MergeBuckets(Int32,Int32)` | Merges two adjacent buckets. |
| `Reset` |  |

