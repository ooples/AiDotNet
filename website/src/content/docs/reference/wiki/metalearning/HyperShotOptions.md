---
title: "HyperShotOptions<T, TInput, TOutput>"
description: "Configuration options for HyperShot (Sendera et al., NeurIPS 2023)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for HyperShot (Sendera et al., NeurIPS 2023).

## For Beginners

HyperShot uses a "network that generates networks":

1. A hypernetwork takes support set features as input
2. It outputs the weights for a classifier tailored to this task
3. The generated classifier is used to classify query examples

This is faster than gradient-based adaptation (like MAML) because it requires
only a single forward pass through the hypernetwork.

## How It Works

HyperShot generates task-specific classifier weights using a hypernetwork conditioned on
support set features. The hypernetwork produces a full classifier in one forward pass.

Reference: Sendera, M., Tabor, J., Nowak, A., & Trzcinski, T. (2023).
HyperShot: Few-Shot Learning by Kernel HyperNetworks. NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperShotOptions(IFullModel<,,>)` | Initializes a new instance of HyperShotOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `HypernetHiddenDim` | Gets or sets the hypernetwork hidden dimension. |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer. |
| `KernelType` | Gets or sets the kernel type for kernel hypernetwork. |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

