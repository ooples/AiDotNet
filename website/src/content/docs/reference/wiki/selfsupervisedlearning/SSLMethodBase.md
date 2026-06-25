---
title: "SSLMethodBase<T>"
description: "Abstract base class for self-supervised learning methods."
section: "API Reference"
---

`Base Classes` · `AiDotNet.SelfSupervisedLearning`

Abstract base class for self-supervised learning methods.
Extends `ModelBase` for unified model framework participation.

## For Beginners

This base class provides common functionality shared by all
SSL methods, including parameter management, training mode control, and configuration handling.

## How It Works

Derived classes (SimCLR, MoCo, BYOL, etc.) implement the specific training logic
in the `SSLAugmentationContext{` method.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SSLMethodBase(INeuralNetwork<>,IProjectorHead<>,SSLConfig)` | Initializes a new instance of the SSLMethodBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `DefaultLossFunction` |  |
| `Name` |  |
| `ParameterCount` |  |
| `Projector` | Gets the projector, throwing if not initialized. |
| `RequiresMemoryBank` |  |
| `UsesMomentumEncoder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePairwiseDistances(Tensor<>)` | Computes the pairwise squared distances between embeddings. |
| `ComputeSimilarityMatrix(Tensor<>,Tensor<>,Boolean)` | Computes similarity matrix between two sets of embeddings. |
| `CosineSimilarity(Tensor<>,Tensor<>)` | Computes cosine similarity between two tensors. |
| `CreateStepResult()` | Creates a default step result with common metrics. |
| `DeepCopy` |  |
| `Encode(Tensor<>)` |  |
| `EncodeAndProject(Tensor<>)` | Encodes input and projects it to the SSL embedding space. |
| `GetAdditionalParameterCount` | Gets the count of additional parameters. |
| `GetAdditionalParameters` | Gets additional parameters specific to this SSL method. |
| `GetEffectiveLearningRate` | Gets the effective learning rate based on configuration and scheduling. |
| `GetEffectiveTemperature` | Gets the effective temperature based on configuration and scheduling. |
| `GetEncoder` |  |
| `GetParameters` |  |
| `L2Normalize(Tensor<>)` | L2-normalizes a tensor along the last dimension. |
| `MatMul(Tensor<>,Tensor<>)` | Computes matrix multiplication with engine-accelerated dot products. |
| `OnEpochEnd(Int32)` | Signals the end of an epoch. |
| `OnEpochStart(Int32)` | Signals the start of a new epoch. |
| `Predict(Tensor<>)` |  |
| `Reset` |  |
| `SetAdditionalParameters(Vector<>,Int32)` | Sets additional parameters specific to this SSL method. |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` | Sets the training mode for the SSL method. |
| `Train(Tensor<>,Tensor<>)` | Trains the SSL method. |
| `TrainStep(Tensor<>,SSLAugmentationContext<>)` |  |
| `TrainStepCore(Tensor<>,SSLAugmentationContext<>)` | Implementation-specific training step logic. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_config` | The SSL configuration. |
| `_currentEpoch` | Current epoch counter. |
| `_currentStep` | Current training step counter. |
| `_encoder` | The main encoder neural network. |
| `_isTraining` | Whether the method is in training mode. |
| `_projector` | The projection head for SSL embeddings. |

