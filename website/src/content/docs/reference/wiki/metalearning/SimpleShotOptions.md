---
title: "SimpleShotOptions<T, TInput, TOutput>"
description: "Configuration options for SimpleShot (Wang et al., 2019) few-shot classification."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for SimpleShot (Wang et al., 2019) few-shot classification.

## For Beginners

SimpleShot shows that simple methods can be surprisingly effective:

**The approach:**

1. Train a good feature extractor on the base classes (standard training, no episodes)
2. For a new few-shot task:
- Extract features for all support and query examples
- Normalize the features (L2 norm or centered L2 norm)
- Compute class centroids from support features
- Classify query examples by nearest centroid

**Why it works:**
Many complex meta-learning methods spend effort on task-specific adaptation, but
a well-trained feature extractor with proper normalization already produces features
where nearest-centroid works well.

**Normalization methods:**

- L2: Normalize each feature vector to unit length
- CL2N: Center features (subtract mean), then L2 normalize
- CL2N typically works better because it removes the "bias" in feature space

## How It Works

SimpleShot demonstrates that nearest-centroid classification with proper feature
normalization (L2 or CL2N) can match or exceed many complex meta-learning methods.
It requires no task-specific adaptation - just normalize features and compare distances.

Reference: Wang, Y., Chao, W.L., Weinberger, K.Q., & van der Maaten, L. (2019).
SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimpleShotOptions(IFullModel<,,>)` | Initializes a new instance of SimpleShotOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `DistanceMetric` | Gets or sets the distance metric for nearest-centroid classification. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer (unused for SimpleShot). |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NormalizationType` | Gets or sets the feature normalization method. |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

