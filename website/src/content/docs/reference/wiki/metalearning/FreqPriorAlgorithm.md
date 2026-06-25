---
title: "FreqPriorAlgorithm<T, TInput, TOutput>"
description: "Implementation of FreqPrior: Frequency-based prior for cross-domain few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of FreqPrior: Frequency-based prior for cross-domain few-shot learning.

## How It Works

FreqPrior decomposes the parameter vector using a discrete cosine transform (DCT)-like
basis. Low-frequency components (capturing smooth, domain-invariant structure) are strongly
regularized toward the meta-prior, while high-frequency components (capturing task-specific
details) are allowed to adapt freely. This encourages learning transferable features
that generalize across domains.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeFrequencyPenalty(List<Vector<>>)` | Computes frequency-based penalty: variance of low-freq gradient components should be small (consistent across tasks), high-freq can vary. |
| `MetaTrain(TaskBatch<,,>)` |  |

