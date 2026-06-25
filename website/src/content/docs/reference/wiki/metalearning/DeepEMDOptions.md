---
title: "DeepEMDOptions<T, TInput, TOutput>"
description: "Configuration options for DeepEMD (Zhang et al., CVPR 2020) few-shot learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for DeepEMD (Zhang et al., CVPR 2020) few-shot learning.

## For Beginners

DeepEMD uses a clever way to compare examples:

**The Problem with simple metrics:**
ProtoNets averages all features into one vector. This loses local details like
"the left part of image A looks like the right part of image B."

**DeepEMD's solution: Earth Mover's Distance**
Think of features as piles of dirt. EMD measures the minimum "work" needed to
reshape one pile into another. This captures structural alignment:

- Each local feature (patch) from one image can match any patch from another
- The cost of matching = distance between the patches
- EMD finds the optimal matching that minimizes total distance

**Three modes:**

- FCN: Feature maps from the last conv layer (spatial patches)
- Grid: Divide feature maps into a regular grid
- Sampling: Sample representative local features

## How It Works

DeepEMD uses the Earth Mover's Distance (optimal transport) as a metric for
comparing image representations in few-shot learning. Unlike ProtoNets which
compares single vectors, DeepEMD compares sets of local features using optimal
transport, capturing fine-grained structural similarity.

Reference: Zhang, C., Cai, Y., Lin, G., & Shen, C. (2020).
DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance and Structured Classifiers. CVPR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepEMDOptions(IFullModel<,,>)` | Initializes a new instance of DeepEMDOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EMDMode` | Gets or sets the EMD mode: "fcn", "grid", or "sampling". |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer. |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `NumNodes` | Gets or sets the number of local features (nodes) per example for EMD computation. |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `SinkhornIterations` | Gets or sets the number of Sinkhorn iterations for approximate EMD. |
| `SinkhornRegularization` | Gets or sets the regularization parameter for Sinkhorn algorithm. |
| `Temperature` | Gets or sets the temperature for the EMD-based classification logits. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

