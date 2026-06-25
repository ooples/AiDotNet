---
title: "BootstrapValidation<T>"
description: "Bootstrap validation for evaluating clustering quality and confidence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Bootstrap validation for evaluating clustering quality and confidence.

## For Beginners

Bootstrap asks "How confident are we in these results?"

By creating many random samples (with repeats allowed):

- We see how much our metrics vary
- We can give confidence intervals, not just single numbers
- We understand which results are reliable

If the bootstrap results vary a lot, be cautious about the clustering!

## How It Works

Bootstrap validation uses resampling with replacement to estimate the
uncertainty in clustering metrics. It provides confidence intervals for
internal validation measures like silhouette score.

Algorithm:

1. Create B bootstrap samples (sample with replacement)
2. Cluster each bootstrap sample
3. Compute metrics on each clustering
4. Use the distribution to estimate confidence intervals

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BootstrapValidation(Int32,Nullable<Int32>)` | Initializes a new BootstrapValidation instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAssignmentConfidence(Matrix<>,Int32)` | Evaluates cluster assignment confidence for each point. |
| `Evaluate(Matrix<>,Int32,Double)` | Evaluates clustering with bootstrap confidence intervals. |
| `EvaluateRange(Matrix<>,Int32,Int32,Double)` | Evaluates bootstrap confidence intervals across a range of cluster counts. |

