---
title: "TabMClassifier<T>"
description: "TabM implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabM implementation for classification tasks.

## For Beginners

Use this model when you want to classify data into categories
with the benefits of an ensemble (multiple models voting).

How predictions work:

1. Input passes through shared layers with per-member modulation
2. Each ensemble member produces class logits
3. Logits are averaged across members
4. Softmax converts to probabilities

Benefits over single models:

- Reduced overfitting through ensemble averaging
- Better calibrated probability estimates
- More robust to noisy features
- Comparable speed to single models (parameter sharing)

Example:

## How It Works

TabMClassifier uses the TabM architecture with BatchEnsemble layers for multi-class
classification. It averages predictions across ensemble members and uses softmax
for probability outputs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabMClassifier` | Initializes a new instance of the TabMClassifier class with default configuration. |
| `TabMClassifier(Int32,Int32,TabMOptions<>)` | Initializes a new instance of the TabMClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of output classes. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySoftmax(Tensor<>)` | Applies softmax to convert logits to probabilities. |
| `ComputeAccuracy(Tensor<>,Int32[])` | Computes classification accuracy. |
| `ComputeCrossEntropyLoss(Tensor<>,Int32[])` | Computes the cross-entropy loss. |
| `ComputeF1Score(Tensor<>,Int32[])` | Computes the macro F1 score across all classes. |
| `Forward(Tensor<>)` | Performs the forward pass to get class logits (per member). |
| `GetParameters` | Gets all parameters including the classification head. |
| `GetPredictiveUncertainty(Tensor<>)` | Gets predictive uncertainty as the entropy of the averaged predictions. |
| `Predict(Tensor<>)` | Predicts the most likely class for each sample. |
| `PredictProbabilities(Tensor<>)` | Predicts class probabilities (averaged across ensemble members). |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Int32[],)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters including the classification head. |

