---
title: "ModelDriftDetector<T>"
description: "Model-based drift detector: uses gradient direction and weight divergence to detect drift."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.DriftDetection`

Model-based drift detector: uses gradient direction and weight divergence to detect drift.

## For Beginners

Instead of monitoring loss values (like StatisticalDriftDetector),
this detector looks at HOW the model is changing. If a client's gradients suddenly point in
a very different direction than before, their data distribution has likely shifted. Similarly,
if their model weights diverge abnormally from the global model, they may be adapting to
a changed local distribution.

## How It Works

**Methods:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelDriftDetector(FederatedDriftOptions)` | Initializes a new instance of `ModelDriftDetector`. |

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

