---
title: "CausalForest<T>"
description: "Causal Forest for heterogeneous treatment effect estimation using random forests."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalInference`

Causal Forest for heterogeneous treatment effect estimation using random forests.

## For Beginners

Causal Forests are like regular random forests, but instead
of predicting outcomes, they predict how much the treatment CHANGES the outcome
for each individual.

Key concepts:

1. CATE (Conditional Average Treatment Effect): The expected treatment effect

for individuals with specific characteristics.

2. Honest estimation: Using separate data for building trees vs estimating effects.
3. Heterogeneity: Treatment effects can vary across the population.

How it works:

1. Build many decision trees on bootstrap samples
2. At each node, split to maximize treatment effect heterogeneity
3. For prediction, average treatment effect estimates across trees

Example interpretation:

- CATE = +5 for young patients: Treatment increases outcome by 5 for young patients
- CATE = -2 for elderly: Treatment decreases outcome by 2 for elderly patients
- This helps target treatments to those who benefit most

References:

- Athey & Imbens (2016). "Recursive Partitioning for Heterogeneous Causal Effects"
- Wager & Athey (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"

## How It Works

Causal Forests extend the random forest framework to estimate Conditional Average
Treatment Effects (CATE) - how treatment effects vary across individuals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CausalForest(Int32,Int32,Int32,Nullable<Int32>,Boolean,Double,Nullable<Int32>)` | Gets the model type. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumTrees` | Gets the number of trees in the forest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AssignToLeaves(CausalForest<>.CausalTree,Matrix<>,List<Int32>,Dictionary<CausalForest<>.CausalTree,List<Int32>>)` | Assigns samples to leaf nodes. |
| `BuildHonestTree(Matrix<>,Vector<Int32>,Vector<>,List<Int32>,List<Int32>,Int32)` | Builds a causal tree using honest estimation. |
| `BuildTree(Matrix<>,Vector<Int32>,Vector<>,List<Int32>,Int32,Int32)` | Builds a causal tree recursively. |
| `CountFeatureSplits(CausalForest<>.CausalTree,Int32[])` | Counts feature splits in a tree. |
| `CreateNewInstance` | Creates a new instance of the same type. |
| `EstimateATE(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect (ATE). |
| `EstimateATT(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect on the Treated (ATT). |
| `EstimateCATEPerIndividual(Matrix<>,Vector<Int32>,Vector<>)` | Estimates CATE for each individual. |
| `EstimateLeafEffect(Vector<Int32>,Vector<>,List<Int32>)` | Estimates the treatment effect in a leaf node. |
| `EstimateLeafEffectDouble(Vector<Int32>,Vector<>,List<Int32>)` | Estimates the treatment effect in a leaf node (double version). |
| `EstimatePropensityScoresCore(Matrix<>)` | Estimates propensity scores. |
| `EstimateTreatmentEffect(Matrix<>)` | Estimates treatment effects for individuals using the causal forest. |
| `FindBestSplit(Matrix<>,Vector<Int32>,Vector<>,List<Int32>,Int32)` | Finds the best split that maximizes treatment effect heterogeneity. |
| `Fit(Matrix<>,Vector<>,Vector<>)` | Fits the causal model using the ICausalModel interface signature. |
| `Fit(Matrix<>,Vector<Int32>,Vector<>)` | Fits the causal forest to the data. |
| `GetFeatureImportance` | Gets feature importance based on split frequency. |
| `GetParameters` | Gets all model parameters. |
| `Predict(Matrix<>)` | Standard prediction - returns treatment effect predictions. |
| `PredictControl(Matrix<>)` | Predicts outcomes under control for the given features. |
| `PredictTreated(Matrix<>)` | Predicts outcomes under treatment for the given features. |
| `PredictTreatmentEffect(Matrix<>)` | Predicts treatment effects for new individuals. |
| `PredictTree(CausalForest<>.CausalTree,Matrix<>,Int32)` | Predicts a single sample through a tree. |
| `ReEstimateLeafEffects(CausalForest<>.CausalTree,Matrix<>,Vector<Int32>,Vector<>,List<Int32>)` | Re-estimates leaf effects using estimation sample (for honest estimation). |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `Shuffle(List<>)` | Shuffles a list in place. |
| `WithParameters(Vector<>)` | Creates a new instance with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedFeatures` | Cached feature matrix from fitting. |
| `_cachedOutcome` | Cached outcome vector from fitting. |
| `_cachedTreatment` | Cached treatment vector from fitting. |
| `_honest` | Whether to use honest estimation (separate data for structure vs estimation). |
| `_honestFraction` | Fraction of data to use for tree building when honest=true. |
| `_maxDepth` | Maximum depth of each tree. |
| `_maxFeatures` | Number of features to consider at each split. |
| `_minSamplesLeaf` | Minimum samples required in a leaf node. |
| `_numTrees` | Number of trees in the forest. |
| `_propensityCoefficients` | Propensity score coefficients for overlap adjustment. |
| `_random` | Random number generator for reproducibility. |
| `_trees` | The trained causal trees. |

