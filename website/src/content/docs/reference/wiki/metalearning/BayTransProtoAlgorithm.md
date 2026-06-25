---
title: "BayTransProtoAlgorithm<T, TInput, TOutput>"
description: "Implementation of BayTransProto: Bayesian Transductive Prototypical Networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of BayTransProto: Bayesian Transductive Prototypical Networks.

## How It Works

BayTransProto combines Bayesian posterior sampling with transductive refinement.
After standard inner-loop adaptation, the algorithm samples multiple parameter vectors
from a Gaussian posterior centered on the adapted params. Transductive refinement
then uses query-set gradients to iteratively update the posterior mean, leveraging
unlabeled query data to improve adaptation.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_logVar` | Learned log-variance for the posterior. |

