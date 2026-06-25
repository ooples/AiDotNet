---
title: "GhostBatchNormalization<T>"
description: "Implements Ghost Batch Normalization, a regularization technique used in TabNet that applies batch normalization to virtual mini-batches within each actual batch."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Implements Ghost Batch Normalization, a regularization technique used in TabNet
that applies batch normalization to virtual mini-batches within each actual batch.

## For Beginners

Batch Normalization helps neural networks train faster by
normalizing the inputs to each layer. Ghost Batch Normalization takes this further
by adding controlled randomness through virtual batches.

Imagine you have a batch of 256 samples:

- Standard Batch Norm: Computes mean/variance over all 256 samples
- Ghost Batch Norm (virtual size 64): Computes 4 separate mean/variance calculations,

one for each group of 64 samples

This variation in statistics acts as regularization, helping prevent overfitting.
It's particularly effective for tabular data where overfitting is common.

## How It Works

Ghost Batch Normalization divides each training batch into smaller "virtual batches"
and computes separate normalization statistics for each. This provides a regularization
effect similar to using smaller batch sizes without the computational overhead.

Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, AAAI 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GhostBatchNormalization(Int32,Int32,Double,Double)` | Initializes a new instance of the GhostBatchNormalization class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta` | Gets the shift (beta) parameters. |
| `Gamma` | Gets the scale (gamma) parameters. |
| `Name` | Gets the name of this layer. |
| `ParameterCount` | Gets the number of trainable parameters in this layer. |
| `RunningMean` | Gets the running mean statistics. |
| `RunningVar` | Gets the running variance statistics. |
| `SupportsTraining` | Gets whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass through the Ghost Batch Normalization layer. |
| `ForwardInference(Tensor<>)` | Performs the forward pass using running statistics (inference mode). |
| `GetOutputShape(Int32[])` | Gets the output shape given an input shape. |
| `GetParameterGradients` | Gets the parameter gradients from the last backward pass. |
| `GetParameters` | Gets the learnable parameters of this layer. |
| `ResetGradients` | Resets the gradients to zero. |
| `SetParameters(Vector<>)` | Sets the learnable parameters of this layer. |
| `SetTrainingMode(Boolean)` | Sets training vs inference mode. |

