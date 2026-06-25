---
title: "LaplacianShotOptions<T, TInput, TOutput>"
description: "Configuration options for LaplacianShot (Ziko et al., ICML 2020) few-shot learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for LaplacianShot (Ziko et al., ICML 2020) few-shot learning.

## For Beginners

LaplacianShot is like "asking your neighbors for help":

**The idea:**

1. Start with nearest-centroid classification (like SimpleShot)
2. Build a graph connecting similar query examples
3. Propagate labels through the graph: if your neighbors are confident about class A,

you should also lean toward class A

**Analogy:**
Imagine voting in a room full of people:

- First, everyone votes independently based on what they see
- Then, people discuss with their nearest neighbors
- After discussion, everyone updates their vote
- The final votes are more accurate because neighbors share information

**Why Laplacian?**
The "Laplacian" is a mathematical way to encode graph smoothness.
Laplacian regularization says: "predictions should be smooth over the graph"
meaning similar examples should have similar predictions.

## How It Works

LaplacianShot extends nearest-centroid classification with Laplacian regularization
over the query set's feature graph. It encourages nearby query examples (in feature space)
to receive similar class assignments, propagating confident predictions to uncertain ones.

Reference: Ziko, I., Dolz, J., Granger, E., & Ben Ayed, I. (2020).
Laplacian Regularized Few-Shot Learning. ICML 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LaplacianShotOptions(IFullModel<,,>)` | Initializes a new instance of LaplacianShotOptions. |

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
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer. |
| `KNearestNeighbors` | Gets or sets the number of nearest neighbors for building the kNN graph. |
| `KernelBandwidth` | Gets or sets the kernel bandwidth for computing graph edge weights. |
| `LaplacianWeight` | Gets or sets the Laplacian regularization weight. |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `PropagationIterations` | Gets or sets the number of label propagation iterations. |
| `RandomSeed` | Gets or sets the random seed. |
| `StepSize` | Gets or sets the step size (alpha) for Laplacian smoothing iterations. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

