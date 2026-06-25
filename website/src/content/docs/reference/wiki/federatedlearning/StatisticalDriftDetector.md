---
title: "StatisticalDriftDetector<T>"
description: "Statistical drift detector: uses Page-Hinkley, ADWIN, or DDM tests on client metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.DriftDetection`

Statistical drift detector: uses Page-Hinkley, ADWIN, or DDM tests on client metrics.

## For Beginners

This detector monitors each client's loss or accuracy over rounds
and applies proven statistical tests to detect when the distribution has shifted.

## How It Works

**Methods:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StatisticalDriftDetector(FederatedDriftOptions)` | Initializes a new instance of `StatisticalDriftDetector`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectDrift(Int32,Dictionary<Int32,Tensor<>>,Tensor<>,Dictionary<Int32,Double>)` |  |
| `GetAdaptiveWeights(Dictionary<Int32,Double>,DriftReport)` |  |
| `Reset` |  |

