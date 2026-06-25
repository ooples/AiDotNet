---
title: "CifAlignmentLayer<T>"
description: "Continuous Integrate-and-Fire (CIF) alignment layer per Gao et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Continuous Integrate-and-Fire (CIF) alignment layer per Gao et al.
2022 "Paraformer" §3.2 / Algorithm 1. Converts a variable-length
encoder hidden-state sequence `[B, S, D]` into a token-aligned
acoustic embedding sequence `[B, S, D]` by predicting per-
timestep fire weights and integrating the hidden states until the
cumulative weight crosses a unit-mass threshold.

## How It Works

**Algorithm (Gao 2022 Algorithm 1):**

**Output shape — fixed [B, S, D]:** the CIF paper's
output length `N` is data-dependent (depends on
`round(Σₜ α_t)`), which doesn't fit a static
`ILayer` shape contract. We follow the FunASR
runtime convention: declare the output as the same length as the
input (a safe upper bound because each α_t ∈ [0, 1] gives at most
one fire per step), populate the first `predicted_N` slots
with the CIF tokens, and zero-pad the remainder. Downstream
attention layers ignore the padded slots through standard
padding-mask handling.

**Trainable parameters:** only the alpha-predictor's
Dense weights. The integrate-and-fire arithmetic itself is
parameter-free and the threshold-crossing is non-differentiable —
gradients flow through the alpha predictor only via the upstream
loss applied to non-firing accumulation paths. Paraformer's
"alpha scaling" training trick (scaling all α_t so their sum
matches the target token count) is the standard way to make the
predictor learn alignment in spite of this; consumers that need
full alignment supervision should apply that scaling at training
time.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CifAlignmentLayer(Int32,Double,Double)` | Initializes a new CIF alignment layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` | Currently `false`: this layer's `Tensor{` materializes α and the integrated hidden states into scalar T values via per-element `Tensor` indexers and scalar `NumOps` arithmetic, which the tape autodiff path cannot record. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `Forward(Tensor<>)` |  |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

