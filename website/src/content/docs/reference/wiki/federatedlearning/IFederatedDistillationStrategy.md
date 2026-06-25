---
title: "IFederatedDistillationStrategy<T>"
description: "Interface for federated knowledge distillation strategies that enable model-heterogeneous FL."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Distillation`

Interface for federated knowledge distillation strategies that enable model-heterogeneous FL.

## For Beginners

Imagine different hospitals using different AI models (some simple,
some complex). Knowledge distillation lets them share what they've learned without requiring
everyone to use the same model. The "knowledge" is transferred through predictions or
summaries rather than model weights.

## How It Works

In standard FL, all clients must use the same model architecture. Federated knowledge
distillation removes this constraint by exchanging knowledge (logits, prototypes, or
generated samples) instead of raw model parameters.

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateKnowledge(Dictionary<Int32,Matrix<>>,Dictionary<Int32,Double>)` | Aggregates knowledge from multiple clients into a global knowledge representation. |
| `ApplyKnowledge(Vector<>,Matrix<>,Matrix<>,Double)` | Applies the aggregated global knowledge to update a local model. |
| `ExtractKnowledge(Vector<>,Matrix<>)` | Generates the knowledge representation to send to the server (logits, prototypes, etc.). |

