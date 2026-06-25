---
title: "SIBAlgorithm<T, TInput, TOutput>"
description: "Implementation of SIB (Sequential Information Bottleneck) for transductive few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of SIB (Sequential Information Bottleneck) for transductive few-shot learning.

## For Beginners

SIB is like an intelligent clustering algorithm:

**How it works:**

1. Start with class prototypes from support examples
2. Assign all query examples to the closest prototype (initial clusters)
3. Iteratively refine:

a. For each example, compute KL divergence to each cluster
b. Reassign to the best cluster (minimum KL divergence)
c. Update cluster statistics
d. Repeat until convergence

4. Multiple restarts ensure we don't get stuck in bad solutions

**Information Bottleneck trade-off:**

- Want to KEEP: Information about which class each example belongs to
- Want to REMOVE: Noise, irrelevant variations, outlier effects
- Beta parameter controls this balance

**Why multiple restarts?**
The SIB optimization landscape can have local optima. Running from different
starting points increases the chance of finding the globally best clustering.

## How It Works

SIB uses the information bottleneck principle to iteratively refine cluster assignments
for transductive few-shot classification. It processes all examples (support + query) jointly,
optimizing a trade-off between information retention and compression.

**Algorithm - SIB:**

Reference: Hu, Y., Gripon, V., & Pateux, S. (2020).
Leveraging the Feature Distribution in Transfer-based Few-Shot Learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SIBAlgorithm(SIBOptions<,,>)` | Initializes a new SIB meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task using the Sequential Information Bottleneck. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step for SIB. |
| `RunSIB(Vector<>,Vector<>)` | Runs the SIB optimization with multiple random restarts. |

