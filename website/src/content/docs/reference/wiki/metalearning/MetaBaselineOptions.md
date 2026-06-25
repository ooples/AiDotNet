---
title: "MetaBaselineOptions<T, TInput, TOutput>"
description: "Configuration options for Meta-Baseline (Chen et al., ICLR 2021)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Meta-Baseline (Chen et al., ICLR 2021).

## For Beginners

Meta-Baseline is a "simple but strong" approach:

**Phase 1: Pre-training**
Train a classifier normally on all base classes (standard cross-entropy).
This produces a strong feature extractor.

**Phase 2: Meta-training**
Fine-tune with episodic training using cosine similarity to class centroids.
This adapts the features for few-shot nearest-centroid classification.

**Why it's effective:**
A well-pretrained backbone provides excellent features. The meta-training
phase just polishes them for the nearest-centroid setting.

## How It Works

Meta-Baseline first pre-trains a feature extractor with standard classification,
then meta-trains with cosine-similarity-based nearest-centroid classification.
This two-phase approach provides a strong baseline that outperforms many complex methods.

Reference: Chen, Y., Liu, Z., Xu, H., Darrell, T., & Wang, X. (2021).
Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning. ICLR 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaBaselineOptions(IFullModel<,,>)` | Initializes a new instance of MetaBaselineOptions. |

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
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `Temperature` | Gets or sets the temperature for cosine similarity scaling. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

