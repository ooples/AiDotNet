---
title: "InContextRLAlgorithm<T, TInput, TOutput>"
description: "Implementation of In-Context RL: meta-RL via in-context learning without explicit gradient updates at test time."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of In-Context RL: meta-RL via in-context learning without
explicit gradient updates at test time.

## How It Works

In-Context RL trains a model to adapt through its forward pass by conditioning on a
growing context buffer. The context buffer stores compressed representations of past
(input, prediction, loss) triplets. A context aggregator (learned attention-like
mechanism) combines the buffer entries into a context vector that modulates the
model's parameters. At test time, no gradients are needed — the model improves
purely by observing more support examples in its context.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_contextEncoderParams` | Context encoder parameters: gradient → context entry. |
| `_modulationParams` | Context-to-parameter modulation projection. |

