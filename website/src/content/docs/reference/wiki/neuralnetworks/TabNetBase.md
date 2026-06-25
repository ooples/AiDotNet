---
title: "TabNetBase<T>"
description: "Base implementation of the TabNet architecture for attentive interpretable tabular learning."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base implementation of the TabNet architecture for attentive interpretable tabular learning.

## For Beginners

TabNet is a neural network designed specifically for tables of data
(like spreadsheets or databases). Unlike traditional neural networks that process all
features at once, TabNet makes decisions in steps:

1. **Step 1**: "Which features should I look at first?"
- Selects a subset of features using sparse attention
- Processes those features through the Feature Transformer

2. **Step 2**: "Based on what I learned, which features next?"
- Uses information from Step 1 to select new features
- Features used before are less likely to be selected again

3. **Steps 3, 4, ...**: Continue refining the decision

At each step, TabNet accumulates knowledge that gets combined into the final output.

This approach provides several benefits:

- **Interpretability**: You can see which features were selected at each step
- **Feature Selection**: Automatically ignores irrelevant features
- **Efficiency**: Doesn't need to process all features for every prediction
- **Performance**: Often matches or beats gradient boosting on tabular data

## How It Works

TabNet is a deep learning architecture specifically designed for tabular data. It uses
sequential attention to choose which features to reason from at each decision step,
enabling both high performance and interpretability.

Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, AAAI 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabNetBase(Int32,Int32,TabNetOptions<>)` | Initializes a new instance of the TabNetBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Hardware-accelerated engine for tensor operations. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `ApplyMask(Tensor<>,Tensor<>)` | Applies attention mask to input features. |
| `ApplyMaskBackward(Tensor<>,Tensor<>)` | Backward pass for mask application. |
| `ApplyReLU(Tensor<>)` | Applies ReLU activation. |
| `ApplyReLUDerivative(Tensor<>,Tensor<>)` | Applies ReLU derivative for backward pass. |
| `ComputeSparsityLoss` | Computes the sparsity regularization loss. |
| `CreateOnes(Int32,Int32)` | Creates a tensor filled with ones. |
| `CreateZeros(Int32,Int32)` | Creates a tensor filled with zeros. |
| `Forward(Tensor<>)` | Performs the forward pass through the TabNet encoder. |
| `GetFeatureImportance` | Gets the feature importance from all decision steps. |
| `GetParameters` | Gets all trainable parameters. |
| `GetStepAttentionMasks` | Gets the attention masks from each decision step. |
| `InitializeDecisionSteps` | Initializes the Feature and Attentive Transformers for each decision step. |
| `InitializeSharedLayers` | Initializes the shared layers used across all decision steps. |
| `ResetState` | Resets the internal state. |
| `SetParameters(Vector<>)` | Sets the trainable parameters. |
| `SetTrainingMode(Boolean)` | Sets training mode. |
| `SplitTensor(Tensor<>,Int32,Int32)` | Splits a tensor along the feature dimension. |
| `UpdateParameters()` | Updates parameters using the specified learning rate. |

## Fields

| Field | Summary |
|:-----|:--------|
| `InputDim` | Number of input features. |
| `NumOps` | Numeric operations provider for generic type T. |
| `Options` | Configuration options for TabNet. |
| `OutputDim` | Number of output values. |

