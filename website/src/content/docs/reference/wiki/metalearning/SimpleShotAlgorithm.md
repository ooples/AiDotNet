---
title: "SimpleShotAlgorithm<T, TInput, TOutput>"
description: "Implementation of SimpleShot for few-shot learning via nearest-centroid classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of SimpleShot for few-shot learning via nearest-centroid classification.

## For Beginners

SimpleShot is the "surprisingly effective baseline":

**How it works:**

1. Train a feature extractor normally (no episodic training needed)
2. For each new task:

a. Extract features from all support examples
b. Normalize features (L2 or centered L2)
c. Compute class centroids (average feature per class)
d. Classify queries by nearest centroid

**Why it matters:**
SimpleShot demonstrates that much of few-shot learning performance comes from
having a good feature extractor, not from complex adaptation mechanisms.
Many SOTA methods' improvements come from better features, not better meta-learning.

**When to use:**

- As a strong baseline before trying complex methods
- When you need fast inference with no adaptation cost
- When you have a good pretrained feature extractor

## How It Works

SimpleShot shows that a well-trained feature extractor combined with simple feature
normalization and nearest-centroid classification can match or exceed many complex
meta-learning algorithms. No inner-loop adaptation is needed.

**Algorithm - SimpleShot:**

Reference: Wang, Y., Chao, W.L., Weinberger, K.Q., & van der Maaten, L. (2019).
SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimpleShotAlgorithm(SimpleShotOptions<,,>)` | Initializes a new SimpleShot meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task using nearest-centroid classification with normalized features. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step for SimpleShot. |
| `NormalizeFeatures(Vector<>)` | Normalizes features using the configured normalization method. |
| `UpdateFeatureMean(TaskBatch<,,>)` | Updates the cached feature mean from the current task batch. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_featureMean` | Cached feature mean for CL2N normalization, computed from training features. |

