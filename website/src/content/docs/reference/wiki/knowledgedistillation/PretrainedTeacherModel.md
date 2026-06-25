---
title: "PretrainedTeacherModel<T>"
description: "Pretrained teacher model from external source (e.g., ImageNet, BERT)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Teachers`

Pretrained teacher model from external source (e.g., ImageNet, BERT).

## How It Works

This wrapper takes a `Func<Vector<T>, Vector<T>>` forward-pass
delegate and invokes it directly on every `Vector{` call.
The wrapper itself performs no caching or graph compilation — any
optimizations (including Tensors' AutoTracer auto-compile) depend entirely
on what happens inside the supplied delegate. A delegate that wraps a
standard neural-network model's `Predict` path will pick up those
engine-level optimizations; a delegate that invokes external code
(pre-converted ONNX, a REST call, etc.) will not.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PretrainedTeacherModel(Func<Vector<>,Vector<>>,Int32,Int32)` | Initializes a new instance using a function delegate (not JIT-compilable). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLogits(Vector<>)` | Gets logits from the pretrained model. |

