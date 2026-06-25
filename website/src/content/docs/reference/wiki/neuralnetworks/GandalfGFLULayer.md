---
title: "GandalfGFLULayer<T>"
description: "GANDALF feature backbone: a stack of Gated Feature Learning Units (GFLUs), the defining component of GANDALF (Joseph & Raj 2022, \"GANDALF: Gated Adaptive Network for Deep Automated Learning of Features\")."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

GANDALF feature backbone: a stack of Gated Feature Learning Units (GFLUs), the defining
component of GANDALF (Joseph & Raj 2022, "GANDALF: Gated Adaptive Network for Deep Automated
Learning of Features").

## How It Works

Each GFLU stage performs (1) **learnable feature selection** — a softmax over a per-stage
learnable weight produces a feature mask that emphasizes a subset of inputs; (2) a **gated
transformation** — the masked features go through a linear layer split into a GLU
(`value ⊙ σ(gate)`); and (3) a **gated residual update** of the running representation
(`h ← g ⊙ glu + (1−g) ⊙ h`), letting later stages build hierarchically on earlier feature
selections. This is what distinguishes GANDALF from plain MLPs (no learnable feature selection)
and from NODE (decision-tree ensembles).

Implemented as one composite layer held in a model's `Layers` list. Per-stage mask weights
are registered trainable tensors; the linear sub-layers are registered via
`ILayer{`; the forward is all tape-recorded Engine ops, so
gradients reach every stage. Output: `[batch, numFeatures]` (a refined feature
representation a linear head maps to the prediction).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GandalfGFLULayer(Int32,Int32)` | Initializes a GFLU stack. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

