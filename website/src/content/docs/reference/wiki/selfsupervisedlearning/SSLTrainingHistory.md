---
title: "SSLTrainingHistory<T>"
description: "Training history from SSL pretraining."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Training history from SSL pretraining.

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomMetrics` | Custom metrics per epoch. |
| `KNNHistory` | k-NN accuracy per epoch (if computed). |
| `LearningRateHistory` | Learning rate per epoch. |
| `LossHistory` | Loss values per epoch. |
| `MomentumHistory` | Momentum value per epoch (for methods with momentum encoder). |
| `StdHistory` | Representation standard deviation per epoch (for collapse detection). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCustomMetric(String,)` | Adds a custom metric value. |
| `AddEpochMetrics(,,Double,Double,Double)` | Adds metrics from a training step. |

