---
title: "MCLAlgorithm<T, TInput, TOutput>"
description: "Implementation of MCL (Meta-learning with Contrastive Learning)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of MCL (Meta-learning with Contrastive Learning).

## For Beginners

MCL teaches features to be useful in TWO ways simultaneously:

**Two complementary objectives:**

1. **Meta-learning loss** (be good at few-shot tasks):

"Given 5 examples of cats and 5 of dogs, classify this query correctly."
This teaches the model HOW to use features for few-shot classification.

2. **Contrastive loss** (organize features well):

"Same-class examples should be close together, different-class far apart."
This teaches features to BE inherently better for comparison.

**Why combine both?**

- Meta-learning alone: Features are good for the task but might not cluster well
- Contrastive learning alone: Features cluster well but might not transfer to new tasks
- Together: Features are well-organized AND transfer well to new few-shot tasks

**Analogy:**
It's like training a librarian (meta-learning = learn to organize books by request)
PLUS organizing books on shelves (contrastive = similar books next to each other).
A librarian who works with well-organized shelves is more efficient.

## How It Works

MCL combines episodic meta-learning with supervised contrastive learning to produce
features that are both discriminative for few-shot tasks and well-clustered in
embedding space.

**Algorithm - MCL:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MCLAlgorithm(MCLOptions<,,>)` | Initializes a new MCL meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using projection + contrastive loss. |
| `ComputeContrastiveLoss(Vector<>,Int32)` | Computes supervised contrastive loss for a set of features. |
| `InitializeProjectionHead` | Initializes the contrastive projection head. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `ProjectFeatures(Vector<>)` | Projects features through the contrastive projection head (2-layer MLP with L2 normalization). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_projectionParams` | Parameters for the contrastive projection head. |

