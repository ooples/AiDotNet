---
title: "RandomSurvivalForest<T>"
description: "Implements Random Survival Forest for survival analysis using ensemble of survival trees."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SurvivalAnalysis`

Implements Random Survival Forest for survival analysis using ensemble of survival trees.

## For Beginners

Random Survival Forest extends Random Forest to handle survival data.
Instead of predicting classes or values, it predicts survival curves. Each tree uses log-rank
statistics to find splits that maximize survival difference between groups.

## How It Works

**How it works:**

- Build many survival trees using bootstrap samples
- At each node, select random features and find the split that maximizes log-rank statistic
- Store survival estimates at each terminal node
- Average survival curves from all trees for prediction

**Key advantages:**

- Handles non-linear relationships and interactions automatically
- Provides variable importance through permutation
- Robust to outliers and doesn't require proportional hazards assumption

**Reference:** Ishwaran et al., "Random Survival Forests" (2008)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomSurvivalForest(Int32,Int32,Int32,Int32,Nullable<Int32>)` | Creates a new Random Survival Forest. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDepth` | Gets the maximum depth of each tree. |
| `MaxFeatures` | Gets the number of features to consider at each split. |
| `MinSamplesLeaf` | Gets the minimum samples per leaf. |
| `NumTrees` | Gets the number of trees in the forest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildTree(Matrix<>,Vector<>,Vector<Int32>,Int32[],Int32)` | Recursively builds a survival tree. |
| `ComputeLogRankStatistic(Vector<>,Vector<Int32>,Int32[],Int32[])` | Computes the log-rank statistic for a split. |
| `CreateLeafNode(Vector<>,Vector<Int32>,Int32[])` | Creates a leaf node with Kaplan-Meier survival estimate. |
| `CreateNewInstance` |  |
| `DeepCopy` | Creates a deep copy of this Random Survival Forest, preserving the trained tree ensemble. |
| `FitSurvivalCore(Matrix<>,Vector<>,Vector<Int32>)` | Fits the Random Survival Forest. |
| `GetBaselineSurvival(Vector<>)` | Gets the baseline survival function. |
| `GetLeafSurvival(RandomSurvivalForest<>.SurvivalTree,Matrix<>,Int32)` | Traverses tree to get leaf survival estimate. |
| `GetParameters` |  |
| `InterpolateLeafSurvival(RandomSurvivalForest<>.SurvivalTree,Double)` | Interpolates survival probability at a specific time from leaf node. |
| `Predict(Matrix<>)` | Predicts median survival time. |
| `PredictHazardRatio(Matrix<>)` | Predicts mortality risk scores (higher = higher risk). |
| `PredictSurvivalProbability(Matrix<>,Vector<>)` | Predicts survival probabilities at specified times. |
| `SetParameters(Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_random` | Random generator for reproducibility. |
| `_trees` | The survival trees. |

