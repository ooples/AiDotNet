---
title: "DKTAlgorithm<T, TInput, TOutput>"
description: "Implementation of DKT (Deep Kernel Transfer) (Patacchiola et al., ICLR 2020)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of DKT (Deep Kernel Transfer) (Patacchiola et al., ICLR 2020).

## For Beginners

DKT pairs a neural network with Gaussian processes:

**Standard approach:**
Neural network extracts features, then a simple classifier (e.g., nearest centroid) classifies.

**DKT's approach:**

1. Neural network extracts features (learns what to compare)
2. GP kernel computes similarity in feature space (learns how to compare)
3. GP provides predictions WITH uncertainty (knows when it's uncertain)

**Why Gaussian Processes?**

- Principled uncertainty: "I'm 90% confident it's a cat" vs "I have no idea"
- Non-parametric: Adapts to any number of support examples
- Kernel-based: The deep kernel captures complex, learned similarities

**The deep kernel:**
Instead of a fixed kernel (like RBF), the kernel operates on learned features:
k(x, x') = k_base(f(x), f(x'))
where f is the neural network and k_base is a standard kernel.
Both are trained end-to-end.

## How It Works

DKT combines deep feature extractors with Gaussian processes. The neural network
learns a feature space in which a GP classifier provides principled Bayesian
predictions with uncertainty estimates.

**Algorithm - DKT:**

**Implementation Notes:**

- MetaTrain uses classification loss as a pragmatic approximation of the GP marginal

log-likelihood. The full GP-based optimization requires matrix inversions per task
which is expensive; the classification loss provides equivalent gradient signal for
training the feature extractor end-to-end.

- The modulation strategy computes a single scalar from GP kernel weights and applies it

uniformly to all backbone parameters. This is a deliberate simplification; per-layer
or per-feature-group modulation could better preserve the GP's learned relationships
but would require additional complexity.

- Adapted models (DKTModel) mutate the shared model instance via SetParameters in Predict.

Callers must ensure single-threaded access to each adapted model instance.

Reference: Patacchiola, M., Turner, J., Crowley, E.J., O'Boyle, M., & Sherron, A. (2020).
Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels. ICLR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DKTAlgorithm(DKTOptions<,,>)` | Initializes a new DKT meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `BuildKernelMatrix(List<Vector<>>)` | Builds the GP kernel matrix between all pairs of feature vectors. |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using the deep kernel GP. |
| `ComputeKernel(Vector<>,Vector<>)` | Computes the RBF kernel value between two feature vectors. |
| `EstimateFeatureDim(Int32,Int32)` | Estimates the per-example feature dimensionality from total support and query lengths. |
| `GPPredict(Vector<>,Vector<>)` | Computes the GP predictive mean for query points given support data. |
| `InitializeKernelParams` | Initializes kernel hyperparameters. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `SplitIntoVectors(Vector<>,Int32,Int32)` | Splits a flat feature vector into a list of per-example multi-dimensional vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_kernelParams` | Learned kernel hyperparameters (length-scale, noise variance). |

