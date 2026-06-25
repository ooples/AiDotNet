---
title: "GANDALFBase<T>"
description: "Base implementation of GANDALF (Gated Additive Neural Decision Forest)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base implementation of GANDALF (Gated Additive Neural Decision Forest).

## For Beginners

GANDALF works like a smart forest of decision trees:

Architecture:

1. **Gating Network**: Learns which features are important
2. **Neural Decision Trees**: Trees with learnable split decisions
3. **Soft Routing**: Samples can go down multiple paths with probabilities
4. **Additive Ensemble**: Tree outputs are summed for final prediction

Key insight: Traditional trees have hard decisions (left or right).
GANDALF uses soft decisions where a sample partially goes both ways,
making the whole thing differentiable and trainable with gradient descent.

Example flow:
Input → Gating (feature importance) → Trees (soft routing) → Sum → Output

## How It Works

GANDALF combines gated feature selection with an ensemble of differentiable
decision trees. Each tree makes soft routing decisions, and their outputs
are combined additively.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GANDALFBase(Int32,GANDALFOptions<>)` | Initializes a new instance of the GANDALFBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Hardware-accelerated engine for tensor operations. |
| `NumTrees` | Gets the number of trees. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `TreeDepth` | Gets the tree depth. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGating(Tensor<>)` | Computes the gating weights (feature importance). |
| `ComputeTreeOutput(Tensor<>,Int32)` | Computes the output of a single tree. |
| `ComputeTreeRouting(Tensor<>,Int32)` | Computes soft routing probabilities through a single tree. |
| `ForwardBackbone(Tensor<>)` | Performs the forward pass through the GANDALF backbone. |
| `GetFeatureImportance(Tensor<>)` | Gets the average feature importance across all predictions. |
| `GetGatingWeights` | Gets the gating weights (feature importance) from the last forward pass. |
| `InitializeNormal(Tensor<>,Double,Random)` | Initializes a tensor with normal distribution. |
| `ResetState` | Resets internal state. |
| `UpdateParameters()` | Updates all parameters using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumFeatures` | Number of input features. |
| `NumOps` | Numeric operations helper for type T. |
| `Options` | The model configuration options. |

