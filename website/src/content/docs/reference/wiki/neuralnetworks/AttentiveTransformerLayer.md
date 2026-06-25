---
title: "AttentiveTransformerLayer<T>"
description: "Implements the Attentive Transformer block used in TabNet architecture for feature selection."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements the Attentive Transformer block used in TabNet architecture for feature selection.

## For Beginners

The Attentive Transformer is the "feature selector" in TabNet.

At each decision step, it answers the question: "Which features should I focus on next?"

Key concepts:

- Takes processed features and a "prior scales" tensor as input
- Prior scales track which features have already been used in previous steps
- Outputs a sparse attention mask (many values are exactly 0)
- Features with high attention values are selected for processing
- Features with 0 attention are completely ignored

This sparse attention provides interpretability - you can see exactly which
features the model considers important for each prediction.

## How It Works

The Attentive Transformer learns which features to pay attention to at each decision step.
It produces sparse attention masks using Sparsemax, ensuring that only the most relevant
features are selected for processing.

The attention mechanism uses:

1. Fully connected layer to compute attention logits
2. Ghost Batch Normalization for regularization
3. Prior scaling to discourage feature reuse
4. Sparsemax to produce sparse probability distribution

Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, AAAI 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AttentiveTransformerLayer(Int32,Int32,Double,Int32,Double,Double)` | Initializes a new instance of the AttentiveTransformer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `ComputeSparsityLoss(Tensor<>)` | Computes the sparsity regularization loss. |
| `Forward(Tensor<>)` | Performs the standard forward pass (without prior scales). |
| `Forward(Tensor<>,Tensor<>)` | Performs the forward pass to compute sparse attention mask. |
| `GetAttentionMask` | Gets the last computed attention mask. |
| `GetBiases` |  |
| `GetInputShape` | Gets the input shape. |
| `GetParameterGradients` | Gets the parameter gradients. |
| `GetParameters` |  |
| `GetWeights` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` | Sets the trainable parameters. |
| `SetTrainingMode(Boolean)` | Sets training mode. |
| `UpdateParameters()` |  |
| `UpdateParameters(Vector<>)` | Updates parameters using the specified parameter values. |
| `UpdatePriorScales(Tensor<>,Tensor<>)` | Updates the prior scales based on the current attention mask. |

