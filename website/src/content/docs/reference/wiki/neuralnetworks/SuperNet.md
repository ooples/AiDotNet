---
title: "SuperNet<T>"
description: "SuperNet implementation for gradient-based neural architecture search (DARTS)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

SuperNet implementation for gradient-based neural architecture search (DARTS).
Implements a differentiable architecture search by maintaining architecture parameters (alpha)
and network weights simultaneously.

## For Beginners

A SuperNet is a "network of all possible networks." It
contains every candidate architecture within a single large network, with learnable
weights that determine which operations are most important. During architecture search,
the SuperNet trains these weights using gradient descent, and the final architecture
is derived by selecting the operations with the highest weights. This is the core
mechanism behind DARTS-style neural architecture search.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SuperNet(SearchSpaceBase<>,Int32,ILossFunction<>)` | Initializes a new SuperNet for differentiable architecture search. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function used by this model for gradient computation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies pre-computed gradients to update the model parameters. |
| `ApplyOperation(Tensor<>,Int32,String)` | Apply a specific operation to input |
| `ApplySoftmax(Matrix<>)` | Apply softmax to architecture parameters |
| `BackwardArchitecture(Tensor<>,Tensor<>)` | Backward pass to compute gradients for architecture parameters |
| `BackwardWeights(Tensor<>,Tensor<>,ILossFunction<>)` | Backward pass to compute gradients for network weights using the specified loss function. |
| `ComputeGradients(Tensor<>,Tensor<>,ILossFunction<>)` | Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters. |
| `ComputeLoss(Tensor<>,Tensor<>)` | Computes mean squared error loss |
| `ComputeLossWithFunction(Tensor<>,Tensor<>,ILossFunction<>)` | Computes loss using the specified loss function. |
| `ComputeTrainingLoss(Tensor<>,Tensor<>)` | Computes training loss for weight updates |
| `ComputeValidationLoss(Tensor<>,Tensor<>)` | Computes validation loss for architecture parameter updates |
| `ConfigureFairness(Vector<Int32>,FairnessMetric[])` | Configures fairness evaluation settings. |
| `DeriveArchitecture` | Derives discrete architecture from continuous parameters (argmax selection) |
| `EnableMethod(InterpretationMethod[])` | Enables specific interpretation methods. |
| `FlattenTensor(Tensor<>)` | Flattens a 2D tensor to a vector. |
| `GenerateTextExplanationAsync(Tensor<>,Tensor<>)` | Generates a text explanation for a prediction. |
| `GetAnchorExplanationAsync(Tensor<>,)` | Gets anchor explanation for a given input. |
| `GetArchitectureGradients` | Gets architecture gradients |
| `GetArchitectureParameters` | Gets architecture parameters for optimization |
| `GetCounterfactualAsync(Tensor<>,Tensor<>,Int32)` | Gets counterfactual explanation for a given input and desired output. |
| `GetFeatureInteractionAsync(Int32,Int32)` | Gets feature interaction effects between two features. |
| `GetGlobalFeatureImportanceAsync(Tensor<>)` | Gets the operation importance for SuperNet architecture search. |
| `GetLimeExplanationAsync(Tensor<>,Int32)` | Gets LIME explanation for a specific input. |
| `GetLocalFeatureImportanceAsync(Tensor<>)` | Gets the local feature importance for a specific input. |
| `GetModelSpecificInterpretabilityAsync` | Gets model-specific interpretability information for SuperNet. |
| `GetOperationName(Int32)` | Gets the human-readable name for a given operation index. |
| `GetPartialDependenceAsync(Vector<Int32>,Int32)` | Gets partial dependence data for specified features. |
| `GetShapValuesAsync(Tensor<>)` | Gets SHAP values for the given inputs. |
| `GetWeightGradients` | Gets weight gradients |
| `GetWeightParameters` | Gets weight parameters for optimization |
| `LoadState(Stream)` | Loads the SuperNet's state (architecture parameters and weights) from a stream. |
| `Predict(Tensor<>)` | Forward pass through the SuperNet with mixed operations |
| `SaveState(Stream)` | Saves the SuperNet's current state (architecture parameters and weights) to a stream. |
| `SetBaseModel(IModel<Tensor<>,Tensor<>,ModelMetadata<>>)` | Sets the base model for interpretability analysis. |
| `Train(Tensor<>,Tensor<>)` | Training is handled externally by alternating architecture and weight updates |
| `ValidateFairnessAsync(Tensor<>,Int32)` | Validates fairness metrics for the given inputs. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultLossFunction` | The default loss function used by this model for gradient computation. |

