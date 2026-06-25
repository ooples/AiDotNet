---
title: "FedAGCContinualLearning<T>"
description: "Implements FedAGC — Adaptive Gradient Correction for federated continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ContinualLearning`

Implements FedAGC — Adaptive Gradient Correction for federated continual learning.

## For Beginners

When a federated model learns new tasks, it tends to forget
old ones (catastrophic forgetting). FedAGC adaptively corrects the gradient during training:
it identifies which gradient directions would harm old task performance and reduces their
magnitude, while allowing gradients that help the new task without hurting old tasks to
pass through freely. The correction strength adapts based on how much conflict exists
between old and new task gradients.

## How It Works

Correction:

Reference: FedAGC: Adaptive Gradient Correction for Federated Continual Learning (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedAGCContinualLearning(Double)` | Creates a new FedAGC strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatedImportance` | Gets the accumulated importance from all previous tasks. |
| `CorrectionStrength` | Gets the correction strength. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateImportance(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ComputeAdaptiveCorrectionStrength(Vector<>,Vector<>)` | Computes adaptive per-parameter correction strength based on how important each parameter is for old tasks vs how much the new task gradient wants to change it. |
| `ComputeImportance(Vector<>,Matrix<>)` |  |
| `ComputeRegularizationPenalty(Vector<>,Vector<>,Vector<>,Double)` |  |
| `ProjectGradient(Vector<>,Vector<>)` |  |
| `ProjectGradientAdaptive(Vector<>,Vector<>,Vector<>)` | Projects gradient using adaptive per-parameter correction strengths. |

