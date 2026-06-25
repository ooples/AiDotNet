---
title: "ActiveLearningOptions"
description: "Represents configuration options for active learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Represents configuration options for active learning.

## For Beginners

Active learning helps when labeling data is expensive or time-consuming.
Instead of randomly selecting samples to label, active learning intelligently picks the samples
that would be most helpful for training the model. This can dramatically reduce the number of
labels needed while achieving similar or better performance.

## How It Works

**Typical Usage:**

**How to Choose a Strategy:**

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the number of samples to select in each active learning iteration. |
| `CommitteeSize` | Gets or sets the number of committee members for QueryByCommittee strategy. |
| `DensityBeta` | Gets or sets the beta parameter for DensityWeightedSampling. |
| `DiversityWeight` | Gets or sets the weight for diversity component in hybrid strategies. |
| `DropoutRate` | Gets or sets the dropout rate for Monte Carlo Dropout in BALD/BatchBALD. |
| `MinimumPoolSize` | Gets or sets the minimum number of samples required in the unlabeled pool. |
| `NormalizeScores` | Gets or sets whether to normalize informativeness scores before selection. |
| `NumClusters` | Gets or sets the number of clusters for diversity-based methods. |
| `NumMcSamples` | Gets or sets the number of Monte Carlo samples for BALD strategy. |
| `NumNeighbors` | Gets or sets the number of nearest neighbors for density estimation. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `Strategy` | Gets or sets the active learning strategy to use. |
| `UncertaintyMeasure` | Gets or sets the uncertainty measure for UncertaintySampling strategy. |
| `UseBatchDiversity` | Gets or sets whether to consider diversity when selecting multiple samples in a batch. |

