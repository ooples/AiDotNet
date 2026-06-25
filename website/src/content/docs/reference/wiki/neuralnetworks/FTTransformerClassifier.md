---
title: "FTTransformerClassifier<T>"
description: "FT-Transformer implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

FT-Transformer implementation for classification tasks.

## For Beginners

Use this model when you want to predict categories
(like spam/not spam, customer segments, product categories, etc.).

How it works:

1. Features are tokenized and processed by transformer layers
2. The [CLS] token captures information from all features
3. A linear layer maps the [CLS] representation to class logits
4. Softmax converts logits to probabilities

Example:

## How It Works

FTTransformerClassifier applies the FT-Transformer architecture to multi-class
classification problems. It uses the [CLS] token output with a linear classification head
and softmax activation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FTTransformerClassifier(Int32,Int32,FTTransformerOptions<>)` | Initializes a new instance of the FTTransformerClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of output classes. |
| `ParameterCount` | Gets the total number of trainable parameters including the classification head. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySoftmax(Tensor<>)` | Applies softmax to convert logits to probabilities. |
| `ComputeAccuracy(Tensor<>,Int32[],Matrix<Int32>)` | Computes classification accuracy. |
| `ComputeCrossEntropyLoss(Tensor<>,Int32[])` | Computes the cross-entropy loss. |
| `ComputeF1Score(Tensor<>,Int32[],Matrix<Int32>)` | Computes the macro F1 score across all classes. |
| `Forward(Tensor<>)` | Performs the forward pass with numerical features only. |
| `Forward(Tensor<>,Matrix<Int32>)` | Performs the forward pass to get class logits. |
| `GetParameters` | Gets all parameters including the classification head. |
| `Predict(Tensor<>)` | Predicts class indices with numerical features only. |
| `Predict(Tensor<>,Matrix<Int32>)` | Predicts the most likely class for each sample. |
| `PredictProbabilities(Tensor<>)` | Predicts class probabilities with numerical features only. |
| `PredictProbabilities(Tensor<>,Matrix<Int32>)` | Predicts class probabilities using softmax. |
| `ResetState` | Resets internal state. |
| `SetParameters(Vector<>)` | Sets all parameters including the classification head. |
| `SoftmaxCrossEntropyLoss(Tensor<>,Int32[])` | Tape-tracked softmax cross-entropy on logits [B, C] against integer targets. |
| `TrainStep(Tensor<>,Int32[],,Matrix<Int32>)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters including the classification head. |

