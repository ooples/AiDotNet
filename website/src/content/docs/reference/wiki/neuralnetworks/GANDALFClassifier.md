---
title: "GANDALFClassifier<T>"
description: "GANDALF implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

GANDALF implementation for classification tasks.

## For Beginners

Use GANDALF for classification when you want:

- Automatic feature importance learning
- Interpretable predictions (can see feature weights and tree paths)
- Good performance on tabular data

Example:

## How It Works

GANDALFClassifier uses gated feature selection with neural decision trees
for multi-class classification. The additive ensemble of trees produces
class logits that are converted to probabilities via softmax.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GANDALFClassifier(Int32,Int32,GANDALFOptions<>)` | Initializes a new instance of the GANDALFClassifier class. |

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

