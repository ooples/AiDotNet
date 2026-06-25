---
title: "MetaPACOHAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-PACOH: Hierarchical PAC-Bayesian Meta-Learning with per-group prior variances."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-PACOH: Hierarchical PAC-Bayesian Meta-Learning with per-group
prior variances.

## How It Works

Meta-PACOH extends PACOH by partitioning the parameter space into G groups and
learning independent prior log-variances {log(σ²_g)} for each group. This allows
different parts of the network (e.g., early vs late layers) to have different levels
of flexibility during task adaptation.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeHierarchicalKL(List<Vector<>>)` | Computes per-group KL divergences between empirical posteriors and group priors. |
| `ComputeHyperPriorKL` | Computes KL divergence from the group log-variances to the hyper-prior. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_groupLogVars` | Per-group prior log-variances. |
| `_groupOf` | Group assignments: _groupOf[d] = group index for parameter d. |
| `_groupSize` | Number of parameters per group. |
| `_priorMean` | Prior mean (meta-learned). |

