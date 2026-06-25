---
title: "IFederatedContinualLearningStrategy<T>"
description: "Interface for federated continual learning strategies that prevent catastrophic forgetting."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.ContinualLearning`

Interface for federated continual learning strategies that prevent catastrophic forgetting.

## For Beginners

Imagine a spam filter that learns about new types of spam each month.
Without continual learning, training on November's spam makes it forget how to catch
October's spam. Federated continual learning lets the global model learn new patterns
while remembering old ones, even though different clients see different evolving data.

## How It Works

In production FL, the data distribution changes over time (new tasks, seasonal shifts,
evolving user behavior). Without continual learning, the global model catastrophically
forgets previously learned knowledge when trained on new data.

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateImportance(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` | Merges importance weights from multiple clients into a global importance estimate. |
| `ComputeImportance(Vector<>,Matrix<>)` | Computes the importance (Fisher information) of each parameter based on current task data. |
| `ComputeRegularizationPenalty(Vector<>,Vector<>,Vector<>,Double)` | Applies the continual learning regularization to prevent forgetting. |
| `ProjectGradient(Vector<>,Vector<>)` | Projects the gradient to be orthogonal to previously important directions. |

