---
title: "TabRClassifier<T>"
description: "TabR implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabR implementation for classification tasks.

## For Beginners

TabR for classification works like having a smart
assistant who remembers all past examples and uses them to help classify
new inputs.

Prediction process:

1. Encode the input features
2. Find similar training samples (neighbors)
3. Use attention to aggregate neighbor information
4. Combine with the encoded input
5. Predict class probabilities

Example:

## How It Works

TabRClassifier uses retrieval-augmented predictions for classification.
It finds similar training samples and uses their information along with
a neural network to make class predictions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabRClassifier` | Initializes a new instance of the TabRClassifier class with default configuration. |
| `TabRClassifier(Int32,Int32,TabROptions<>)` | Initializes a new instance of the TabRClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of output classes. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAccuracy(Tensor<>,Int32[])` | Computes classification accuracy. |
| `ComputeCrossEntropyLoss(Tensor<>,Int32[])` | Computes the cross-entropy loss. |
| `Forward(Tensor<>,Vector<Int32>)` | Performs the forward pass to get class logits. |
| `GetPredictionExplanations(Tensor<>)` | Gets interpretability information: which neighbors influenced each prediction. |
| `Predict(Tensor<>)` | Predicts the most likely class for each sample. |
| `PredictProbabilities(Tensor<>)` | Predicts class probabilities. |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Int32[],,Vector<Int32>)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters. |

