---
title: "Sparsemax<T>"
description: "Implements the Sparsemax activation function, which projects input onto the probability simplex with sparse outputs (many exact zeros)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Implements the Sparsemax activation function, which projects input onto the probability simplex
with sparse outputs (many exact zeros).

## For Beginners

Sparsemax is like softmax but produces cleaner, more focused attention.

Imagine you're selecting which features to pay attention to:

- Softmax says "pay a little attention to everything"
- Sparsemax says "focus only on the important features, ignore the rest completely"

This makes neural networks more interpretable because you can clearly see
which features the model considers important (non-zero values) versus
which ones it ignores (exact zeros).

In TabNet, sparsemax is used to select which features to use at each decision step,
providing built-in interpretability about feature importance.

## How It Works

Sparsemax is an alternative to softmax that produces sparse probability distributions.
Unlike softmax, which always produces non-zero probabilities for all inputs, sparsemax
can produce exact zeros, making it ideal for feature selection in attention mechanisms.

**Mathematical Background:**
Sparsemax solves: argmin_{p ∈ Δ^K} ||p - z||²
where Δ^K is the (K-1)-dimensional probability simplex.

Reference: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
by André F. T. Martins and Ramón Fernandez Astudillo (ICML 2016)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Sparsemax` | Initializes a new instance of the Sparsemax class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFlatIndex(Int32[],Int32[])` | Computes flat array index from multi-dimensional indices. |
| `Forward(Tensor<>,Int32)` | Applies the sparsemax function to the input tensor along the specified axis. |
| `ProcessAlongAxis(Tensor<>,Tensor<>,Int32,Int32)` | Processes the input tensor along the specified axis to apply sparsemax. |
| `ProcessGradientAlongAxis(Tensor<>,Tensor<>,Tensor<>,Int32)` | Processes gradient along the specified axis. |
| `ProcessHigherDimensionalAxis(Tensor<>,Tensor<>,Int32)` | Handles sparsemax for tensors with rank > 2. |
| `ProcessHigherDimensionalGradient(Tensor<>,Tensor<>,Tensor<>,Int32)` | Handles gradient computation for tensors with rank > 2. |
| `SparsemaxGradientVector(Vector<>,Vector<>)` | Computes the gradient for a single sparsemax slice. |
| `SparsemaxVector(Vector<>)` | Applies sparsemax to a 1D vector (single slice). |

