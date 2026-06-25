---
title: "FeatureTransformerLayer<T>"
description: "Implements the Feature Transformer block used in TabNet architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements the Feature Transformer block used in TabNet architecture.

## For Beginners

The Feature Transformer is like a smart processor that takes selected
features and transforms them into useful representations.

Key concepts:

- **Shared Layers**: Parameters shared across all decision steps (helps learn common patterns)
- **Step-Specific Layers**: Parameters unique to each step (learns step-specific patterns)
- **GLU (Gated Linear Unit)**: A gating mechanism that controls information flow
- **Residual Connections**: Helps with gradient flow during training

Think of it as a two-part filter:

1. One part decides what information to keep (the "gate")
2. The other part provides the actual information
3. The final output is the product of both

This architecture allows TabNet to learn complex feature interactions effectively.

## How It Works

The Feature Transformer processes selected features at each decision step using shared and
step-specific layers. It employs a GLU (Gated Linear Unit) mechanism for non-linear transformations
combined with Ghost Batch Normalization for regularization.

Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, AAAI 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureTransformerLayer(Int32,Int32,List<FullyConnectedLayer<>>,List<GhostBatchNormalization<>>,Int32,Int32,Int32,Double,Double)` | Initializes a new instance of the FeatureTransformer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SharedBNLayers` | Gets the shared batch normalization layers for reuse. |
| `SharedFCLayers` | Gets the shared fully connected layers for reuse in other FeatureTransformers. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGLU(Tensor<>)` | Applies the GLU (Gated Linear Unit) activation. |
| `ClearGradients` | Clears accumulated gradients. |
| `Forward(Tensor<>)` | Performs the forward pass through the Feature Transformer. |
| `GetBiases` | Gets the biases tensor (not applicable for this composite layer). |
| `GetInputShape` | Gets the input shape for this layer. |
| `GetParameterGradients` | Gets the parameter gradients from the last backward pass. |
| `GetParameters` |  |
| `GetWeights` | Gets the weights tensor (not applicable for this composite layer). |
| `InitializeSharedLayers` | Initializes the shared layers. |
| `InitializeStepSpecificLayers` | Initializes the step-specific layers. |
| `ResetState` |  |
| `SetParameters(Vector<>)` | Sets the trainable parameters of this layer. |
| `SetTrainingMode(Boolean)` | Sets training mode. |
| `UpdateParameters()` |  |
| `UpdateParameters(Vector<>)` | Updates the parameters using the specified parameter values. |

