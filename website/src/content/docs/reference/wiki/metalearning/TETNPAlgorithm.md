---
title: "TETNPAlgorithm<T, TInput, TOutput>"
description: "Translation-Equivariant Transformer Neural Process (TE-TNP, 2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Translation-Equivariant Transformer Neural Process (TE-TNP, 2024).
Combines TNP's transformer attention with relative positional encoding so that
predictions are equivariant to translations of the input space.

## How It Works

**Key Idea:** Standard TNP uses absolute positions in attention. TE-TNP replaces
position keys/queries with sinusoidal encodings of the *relative* displacement between
context and target points: PE(x_i - x_j). This makes the mapping equivariant to input translations.

**Algorithm:**

- Encode each context pair (x, y) into a representation r_i via the base encoder.
- Compute pairwise relative position encodings between all context points.
- Apply multi-head self-attention using relative positional keys to refine representations.
- Aggregate refined representations and modulate model parameters.

**Reference:** Gridded Transformer Neural Processes for Large Unstructured Spatio-Temporal Data (2024).

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ApplyRelativeSelfAttention(List<Vector<>>,Vector<>,Vector<>)` | Applies multi-head self-attention with relative positional encoding. |
| `ComputeEquivarianceReg(List<Vector<>>,List<Vector<>>)` | Equivariance regularization: penalizes variance in representation norms to encourage consistent behavior under translations. |
| `MetaTrain(TaskBatch<,,>)` |  |

