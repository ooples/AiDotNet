---
title: "AutoIntClassifier<T>"
description: "AutoInt implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

AutoInt implementation for classification tasks.

## For Beginners

Use AutoInt for classification when:

- Feature interactions matter (e.g., age + product_category)
- You don't want to manually engineer feature crosses
- You want interpretable interaction patterns

Example:

## How It Works

AutoIntClassifier uses multi-head self-attention to automatically learn
feature interactions for classification on tabular data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoIntClassifier(Int32,Int32,AutoIntOptions<>)` | Initializes a new instance of the AutoIntClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of output classes. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAccuracy(Tensor<>,Vector<Int32>,Matrix<Int32>)` | Computes classification accuracy. |
| `ComputeCrossEntropyLoss(Tensor<>,Vector<Int32>)` | Computes the cross-entropy loss. |
| `Forward(Tensor<>,Matrix<Int32>)` | Performs the forward pass to get class logits. |
| `Predict(Tensor<>,Matrix<Int32>)` | Predicts the most likely class for each sample. |
| `PredictProbabilities(Tensor<>,Matrix<Int32>)` | Predicts class probabilities using softmax. |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Vector<Int32>,,Matrix<Int32>)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters. |

