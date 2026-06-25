---
title: "TabNetClassifier<T>"
description: "TabNet implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabNet implementation for classification tasks.

## For Beginners

Use TabNetClassifier when you want to categorize data into
different classes (groups) from tabular data.

Example use cases:

- Customer churn prediction (will stay / will leave)
- Fraud detection (legitimate / fraudulent transaction)
- Medical diagnosis (healthy / disease A / disease B / ...)
- Credit risk assessment (low / medium / high risk)

Key features:

- **Automatic Feature Selection**: Learns which features matter for classification
- **Interpretability**: See which features drove each classification decision
- **Multi-class Support**: Works for binary (2 classes) or multi-class problems
- **Probability Outputs**: Returns confidence scores for each class

Basic usage:

## How It Works

TabNetClassifier extends the TabNet architecture for predicting discrete class labels.
It uses softmax activation on the output layer to produce class probabilities.

Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, AAAI 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabNetClassifier(Int32,Int32,TabNetOptions<>)` | Initializes a new instance of the TabNetClassifier class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySoftmax(Tensor<>)` | Applies softmax activation to convert logits to probabilities. |
| `ComputeAccuracy(Int32[],Tensor<>)` | Computes the accuracy (percentage of correct predictions). |
| `ComputeCrossEntropyGradient(Tensor<>,Tensor<>)` | Computes the gradient of cross-entropy loss with softmax. |
| `ComputeCrossEntropyGradientOneHot(Tensor<>,Tensor<>)` | Computes the gradient with one-hot encoded targets. |
| `ComputeCrossEntropyLoss(Tensor<>,Tensor<>)` | Computes the cross-entropy loss for classification. |
| `ComputeCrossEntropyLossOneHot(Tensor<>,Tensor<>)` | Computes cross-entropy loss with one-hot encoded targets. |
| `ComputeF1Score(Int32[],Tensor<>,Int32)` | Computes the F1 score for binary classification. |
| `ComputeTotalLoss(Tensor<>,Tensor<>)` | Computes the total loss including sparsity regularization. |
| `Forward(Tensor<>)` | Performs forward pass with softmax activation. |
| `ForwardLogits(Tensor<>)` | Gets the raw logits without softmax. |
| `Predict(Tensor<>)` | Predicts class labels (indices of highest probability classes). |
| `PredictProbabilities(Tensor<>)` | Gets class probabilities for each sample. |
| `TrainStep(Tensor<>,Tensor<>,)` | Performs a single training step. |

