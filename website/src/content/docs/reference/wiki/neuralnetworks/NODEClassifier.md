---
title: "NODEClassifier<T>"
description: "NODE implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

NODE implementation for classification tasks.

## For Beginners

Use NODE for classification when you want:

- Interpretable models (see feature importance via GetFeatureImportance())
- Tree-based structure with neural network trainability
- Good performance on tabular data with mixed feature types

Example:

## How It Works

NODEClassifier uses an ensemble of differentiable oblivious decision trees
for multi-class classification on tabular data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NODEClassifier(Int32,Int32,NODEOptions<>)` | Initializes a new instance of the NODEClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of output classes. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAccuracy(Tensor<>,Vector<Int32>)` | Computes classification accuracy. |
| `ComputeCrossEntropyLoss(Tensor<>,Vector<Int32>)` | Computes the cross-entropy loss. |
| `Forward(Tensor<>)` | Performs the forward pass to get class logits. |
| `Predict(Tensor<>)` | Predicts the most likely class for each sample. |
| `PredictProbabilities(Tensor<>)` | Predicts class probabilities using softmax. |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Vector<Int32>,)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters. |

