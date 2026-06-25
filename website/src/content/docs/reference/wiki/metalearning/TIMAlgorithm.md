---
title: "TIMAlgorithm<T, TInput, TOutput>"
description: "Implementation of TIM (Transductive Information Maximization) for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of TIM (Transductive Information Maximization) for few-shot learning.

## For Beginners

TIM lets query examples help classify each other:

**The key insight:**
If you're classifying 15 query examples into 5 classes, you know each class
should get roughly 3 examples. TIM uses this constraint along with confidence
maximization to iteratively refine predictions.

**How it works:**

1. Start with initial predictions (e.g., nearest centroid from support set)
2. Iteratively refine by optimizing:
- Each query should be confidently assigned to ONE class (low conditional entropy)
- Classes should be balanced overall (high marginal entropy)
- Don't deviate too far from initial predictions
3. After convergence, output the refined predictions

**Why it works:**
By processing queries together, TIM avoids "lonely" misclassifications.
If most queries near a centroid say "class A", the outlier gets pulled in too.

## How It Works

TIM is a transductive few-shot method that refines query predictions by maximizing
mutual information between features and predicted labels. It processes all query
examples jointly, using the query set's structure for better classification.

**Algorithm - TIM:**

Reference: Boudiaf, M., Ziko, I., Rony, J., Dolz, J., Piantanida, P., & Ben Ayed, I. (2020).
Information Maximization for Few-Shot Learning. NeurIPS 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TIMAlgorithm(TIMOptions<,,>)` | Initializes a new TIM meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task using transductive information maximization. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step for TIM. |
| `TransductiveRefine(Vector<>,IMetaLearningTask<,,>)` | Performs transductive refinement by maximizing mutual information. |

