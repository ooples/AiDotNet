---
title: "DKTOptions<T, TInput, TOutput>"
description: "Configuration options for DKT (Deep Kernel Transfer) (Patacchiola et al., ICLR 2020)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for DKT (Deep Kernel Transfer) (Patacchiola et al., ICLR 2020).

## For Beginners

DKT combines neural networks with Gaussian processes:

1. A neural network extracts features (like other meta-learners)
2. A Gaussian Process (GP) classifier operates on these features
3. The GP provides not just predictions but confidence/uncertainty estimates
4. The kernel function (similarity measure) is learned end-to-end

Think of it as: the neural network finds the right feature space,
and the GP provides principled probabilistic classification in that space.

## How It Works

DKT combines deep feature extractors with Gaussian processes for Bayesian few-shot
classification. The deep features serve as the kernel input space, and the GP provides
principled uncertainty estimates.

Reference: Patacchiola, M., Turner, J., Crowley, E.J., O'Boyle, M., & Sherron, A. (2020).
Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels. ICLR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DKTOptions(IFullModel<,,>)` | Initializes a new instance of DKTOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `FeatureDim` | Gets or sets the per-example feature dimension for splitting flattened vectors. |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer. |
| `KernelLengthScale` | Gets or sets the kernel length-scale parameter. |
| `KernelType` | Gets or sets the kernel type for the Gaussian process. |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NoiseVariance` | Gets or sets the noise variance for the GP likelihood. |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

