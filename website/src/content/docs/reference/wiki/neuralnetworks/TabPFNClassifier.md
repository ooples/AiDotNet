---
title: "TabPFNClassifier<T>"
description: "TabPFN implementation for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabPFN implementation for classification tasks.

## For Beginners

TabPFN classification works differently:

1. First, call SetContext() with your training data
2. Then, call Predict() with test data
3. The model uses attention to "learn" from context
4. No training loop needed - it's instant!

Example:

## How It Works

TabPFNClassifier uses in-context learning for tabular classification.
It takes training data as context and makes predictions on test data
in a single forward pass, similar to how language models complete text.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabPFNClassifier(Int32,Int32,TabPFNOptions<>)` | Initializes a new instance of the TabPFNClassifier class. |

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
| `PredictEnsemble(Tensor<>,Int32,Matrix<Int32>)` | Predicts with ensemble averaging over multiple permutations. |
| `PredictProbabilities(Tensor<>,Matrix<Int32>)` | Predicts class probabilities using softmax. |
| `ResetState` | Resets internal state. |
| `SetContext(Tensor<>,Vector<Int32>)` | Sets the context (training) data for in-context learning. |
| `TrainStep(Tensor<>,Vector<Int32>,,Matrix<Int32>)` | Performs a single training step (for fine-tuning). |
| `UpdateParameters()` | Updates all parameters. |

