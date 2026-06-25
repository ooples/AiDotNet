---
title: "MambularClassifier<T>"
description: "Mambular implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Mambular implementation for classification tasks.

## For Beginners

Use Mambular for classification when:

- You have many features and need efficient processing
- You want an alternative to transformer-based models
- Feature order/sequence matters for your task

Example:

## How It Works

MambularClassifier applies State Space Models (Mamba architecture) to
tabular data for multi-class classification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MambularClassifier(Int32,Int32,MambularOptions<>)` | Initializes a new instance of the MambularClassifier class. |

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

