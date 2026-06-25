---
title: "DistillationLoss<T>"
description: "DistillationLoss<T> — Models & Types in AiDotNet.KnowledgeDistillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistillationLoss(Double,Double)` | Initializes a new instance of the DistillationLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes the gradient of the distillation loss for backpropagation. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes the combined distillation loss (soft loss from teacher + hard loss from true labels). |
| `CrossEntropy(Matrix<>,Matrix<>)` | Computes cross-entropy loss: H(true_labels, predictions) = -sum(true_labels * log(predictions)). |
| `KLDivergence(Matrix<>,Matrix<>)` | Computes Kullback-Leibler divergence: KL(p \|\| q) = sum(p * log(p / q)). |
| `Softmax(Matrix<>,Double)` | Applies softmax function with temperature scaling to convert logits to probabilities. |

