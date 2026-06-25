---
title: "PMFAlgorithm<T, TInput, TOutput>"
description: "Implementation of PMF (P>M>F: Pre-training, Meta-training, Fine-tuning)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of PMF (P>M>F: Pre-training, Meta-training, Fine-tuning).

## For Beginners

PMF combines the best of three worlds:

**Pre-training:** Learn general features from lots of data (foundation)
**Meta-training:** Learn to adapt quickly from few examples (specialization)
**Fine-tuning:** Squeeze out extra accuracy on each test task (polish)

This simple pipeline achieves remarkable results because each stage
addresses a different aspect of few-shot learning.

## How It Works

PMF implements a three-stage training pipeline:

1. Pre-training: Standard supervised training on base classes
2. Meta-training: Episodic training for few-shot adaptation
3. Fine-tuning: Optional per-task fine-tuning during adaptation

Reference: Hu, S.X., Li, D., Stuhmer, J., Kim, M., & Hospedales, T.M. (2022).
Pushing the Limits of Simple Pipelines for Few-Shot Learning. ICLR 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PMFAlgorithm(PMFOptions<,,>)` | Initializes a new PMF meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task with optional fine-tuning (Stage 3: F). |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step (Stage 2: M). |

