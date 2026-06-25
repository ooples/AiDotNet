---
title: "TransformerTeacherModel<T>"
description: "Transformer-based teacher model that provides logits from transformer architectures."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Teachers`

Transformer-based teacher model that provides logits from transformer architectures.

## How It Works

This wrapper takes a `Func<Vector<T>, Vector<T>>` forward-pass
delegate and invokes it directly on every `Vector{` call.
The wrapper performs no caching or graph compilation itself — any
optimizations (including Tensors' AutoTracer auto-compile) depend on what
the supplied delegate does internally.

For attention-based distillation strategies that need attention weights, implement
a custom IDistillationStrategy that can extract attention from the underlying model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerTeacherModel(Func<Vector<>,Vector<>>,Int32,Int32)` | Initializes a new instance of the TransformerTeacherModel class using a function delegate. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLogits(Vector<>)` | Gets logits from the transformer model. |

