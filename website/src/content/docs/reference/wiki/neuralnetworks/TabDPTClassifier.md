---
title: "TabDPTClassifier<T>"
description: "TabDPT implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabDPT implementation for classification tasks.

## For Beginners

Use TabDPT for classification when:

- You have a standard tabular classification problem
- You want to leverage foundation model capabilities
- Your data has complex feature interactions

Example:

## How It Works

TabDPTClassifier applies foundation model concepts to tabular classification,
leveraging pre-trained representations for multi-class prediction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabDPTClassifier(Int32,Int32,TabDPTOptions<>)` | Initializes a new instance of the TabDPTClassifier class. |

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

