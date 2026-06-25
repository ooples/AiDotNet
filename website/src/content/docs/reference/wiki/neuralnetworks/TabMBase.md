---
title: "TabMBase<T>"
description: "Base implementation of TabM, a parameter-efficient ensemble model for tabular data."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base implementation of TabM, a parameter-efficient ensemble model for tabular data.

## For Beginners

TabM is a neural network that combines the power of ensemble
methods (multiple models voting) with the efficiency of parameter sharing.

Architecture overview:

1. **Input Layer**: Raw features or embedded features
2. **Hidden Layers**: BatchEnsemble layers with shared weights + per-member modulation
3. **Output Layer**: Predictions for each ensemble member
4. **Aggregation**: Average (or other) of member predictions

Why TabM works well:

- Ensembles reduce variance and improve generalization
- Parameter sharing keeps the model small and efficient
- Diverse members (via rank vectors) capture different aspects of the data
- Training is stable and convergence is fast

Comparison to other methods:

- vs Gradient Boosting: Often similar performance, TabM is more parallelizable
- vs Random Forests: TabM learns feature interactions, RF doesn't
- vs Deep Ensembles: TabM is 4-10x more parameter efficient

## How It Works

TabM uses BatchEnsemble-style parameter sharing to create multiple ensemble members
with minimal parameter overhead. The architecture consists of:

1. Optional feature embedding layer
2. Multiple BatchEnsemble hidden layers with activation and normalization
3. Prediction head

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabMBase(Int32,TabMOptions<>)` | Initializes a new instance of the TabMBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Hardware-accelerated engine for tensor operations. |
| `HiddenDimensions` | Gets the hidden layer dimensions. |
| `NumMembers` | Gets the number of ensemble members. |
| `ParameterCount` | Gets the total number of trainable parameters in the base model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AverageMemberOutputs(Tensor<>,Int32)` | Averages predictions across ensemble members. |
| `ForwardBackbone(Tensor<>)` | Performs the forward pass through the TabM backbone. |
| `GetEnsembleDiversity(Tensor<>,Int32)` | Gets the diversity of ensemble predictions (standard deviation across members). |
| `GetLastHiddenDim` | Gets the output dimension of the last hidden layer. |
| `GetParameterGradients` | Gets parameter gradients as a single vector. |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `InitializeXavier(Tensor<>)` | Initializes a tensor with Xavier/Glorot initialization. |
| `ResetState` | Resets internal state. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a vector. |
| `UpdateParameters()` | Updates all parameters using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumFeatures` | Number of input features. |
| `NumOps` | Numeric operations helper for type T. |
| `Options` | The model configuration options. |

