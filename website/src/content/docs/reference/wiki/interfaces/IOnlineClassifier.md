---
title: "IOnlineClassifier<T>"
description: "Interface for online (incremental) classification models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for online (incremental) classification models.

## For Beginners

Online classifiers can learn from data one sample at a time,
without needing to retrain from scratch. This is useful for:

- Streaming data where samples arrive continuously
- Large datasets that don't fit in memory
- Applications requiring real-time adaptation

## How It Works

**Key differences from batch learning:**

- No need to store all training data
- Model updates incrementally with each sample
- Can adapt to concept drift
- Trade-off: may not achieve same accuracy as batch training

**Common online learning algorithms:**

- Hoeffding Tree (Very Fast Decision Tree)
- Online Naive Bayes
- Stochastic Gradient Descent
- Perceptron

## Properties

| Property | Summary |
|:-----|:--------|
| `IsWarm` | Gets whether the model is warm (has seen at least one sample). |
| `SamplesSeen` | Gets the total number of samples the model has seen. |

## Methods

| Method | Summary |
|:-----|:--------|
| `PartialFit(Matrix<>,Vector<>)` | Updates the model with a batch of training samples. |
| `PartialFit(Vector<>,)` | Updates the model with a single training sample. |

