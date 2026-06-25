---
title: "SAINTClassifier<T>"
description: "SAINT implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

SAINT implementation for classification tasks.

## For Beginners

Use SAINT for classification when:

- You have tabular data with mixed numerical and categorical features
- You believe similar samples in your data share useful patterns
- You want state-of-the-art performance on tabular classification

Example:

## How It Works

SAINTClassifier applies both column attention (over features) and row attention
(over samples in a batch) for multi-class classification on tabular data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAINTClassifier(Int32,Int32,SAINTOptions<>)` | Initializes a new instance of the SAINTClassifier class. |

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

