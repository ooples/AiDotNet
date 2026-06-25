---
title: "PMFOptions<T, TInput, TOutput>"
description: "Configuration options for PMF (P>M>F: Pre-training, Meta-training, Fine-tuning) (Hu et al., ICLR 2022)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for PMF (P>M>F: Pre-training, Meta-training, Fine-tuning) (Hu et al., ICLR 2022).

## For Beginners

PMF is a three-step recipe for few-shot learning:

**Stage 1: Pre-training (P)**
Train a powerful feature extractor on a large dataset (like ImageNet).
This gives a strong foundation of general visual knowledge.

**Stage 2: Meta-training (M)**
Fine-tune with episodic meta-learning on the few-shot task distribution.
This adapts the pretrained features for few-shot scenarios.

**Stage 3: Fine-tuning (F)**
Optionally fine-tune on each test task's support set for extra performance.

**Why it works:**
Pre-training provides rich, transferable features. Meta-training adapts them
for few-shot scenarios. Fine-tuning squeezes out the last bit of accuracy.
Each stage builds on the previous one.

## How It Works

PMF introduces a three-stage training pipeline that leverages large-scale pretraining
before meta-learning. The key insight is that combining pre-training with meta-training
and optional fine-tuning achieves state-of-the-art few-shot performance.

Reference: Hu, S.X., Li, D., Stuhmer, J., Kim, M., & Hospedales, T.M. (2022).
Pushing the Limits of Simple Pipelines for Few-Shot Learning. ICLR 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PMFOptions(IFullModel<,,>)` | Initializes a new instance of PMFOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `DistanceMetric` | Gets or sets the distance metric for classification. |
| `EnableCheckpointing` |  |
| `EnableFineTuning` | Gets or sets whether to use fine-tuning during adaptation (Stage 3). |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `FineTuningLearningRate` | Gets or sets the fine-tuning learning rate for Stage 3 (F). |
| `FineTuningSteps` | Gets or sets the number of fine-tuning steps during adaptation. |
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
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

