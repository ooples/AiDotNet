---
title: "SSLMetrics<T>"
description: "Metrics for monitoring and evaluating self-supervised learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Evaluation`

Metrics for monitoring and evaluating self-supervised learning.

## For Beginners

These metrics help track the quality of SSL training
and detect potential issues like representation collapse. Monitoring these
during training helps ensure the model is learning useful representations.

## How It Works

**Key metrics:**

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAlignment(Tensor<>,Tensor<>)` | Computes alignment loss between positive pairs. |
| `ComputeCosineSimilarity(Tensor<>,Tensor<>)` | Computes cosine similarity between corresponding pairs. |
| `ComputeEffectiveRank(Tensor<>)` | Computes the effective rank of the representation matrix. |
| `ComputeEigenvaluesJacobi(Tensor<>,Int32,Int32)` | Computes eigenvalues of a symmetric matrix using the Jacobi eigenvalue algorithm. |
| `ComputeFullReport(Tensor<>,Tensor<>)` | Computes a full set of SSL metrics. |
| `ComputeRepresentationStd(Tensor<>)` | Computes the standard deviation of representations (collapse detection). |
| `ComputeUniformity(Tensor<>,Double)` | Computes uniformity loss (how uniformly distributed embeddings are). |
| `DetectCollapse(Tensor<>,Double)` | Detects if representations are collapsing. |

