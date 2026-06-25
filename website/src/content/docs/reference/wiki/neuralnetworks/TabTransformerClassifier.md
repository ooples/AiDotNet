---
title: "TabTransformerClassifier<T>"
description: "TabTransformer implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabTransformer implementation for classification tasks.

## For Beginners

Use TabTransformer for classification when:

- You have important categorical features (like sector, region, category)
- You believe there are interactions between categorical features
- You want the model to learn these interactions automatically

Example:

## How It Works

TabTransformerClassifier applies transformer attention to categorical features
for multi-class classification. Numerical features are concatenated after
the categorical embeddings are transformed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabTransformerClassifier(Int32,Int32,TabTransformerOptions<>)` | Initializes a new instance of the TabTransformerClassifier class. |

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

