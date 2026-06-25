---
title: "InContextRLOptions<T, TInput, TOutput>"
description: "Configuration options for In-Context RL: meta-RL via in-context adaptation without explicit gradient updates at test time."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for In-Context RL: meta-RL via in-context adaptation
without explicit gradient updates at test time.

## How It Works

In-Context RL trains a model to perform RL adaptation purely through its forward pass,
without any gradient updates at test time. The model conditions on a context buffer
of past (input, output, loss) triplets and learns to improve its predictions based
on this growing context. During meta-training, the context is built sequentially.

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextBufferSize` | Maximum size of the in-context buffer. |
| `ContextEmbeddingDim` | Dimensionality of context embeddings per entry. |
| `ContextPredictionWeight` | Weight for the context prediction loss (auxiliary). |

