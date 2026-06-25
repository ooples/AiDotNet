---
title: "RandomForestClassifier<T>"
description: "Random Forest classifier that combines multiple decision trees trained on random subsets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Ensemble`

Random Forest classifier that combines multiple decision trees trained on random subsets.

## For Beginners

Random Forest is one of the most popular and powerful machine learning algorithms.
It works by creating a "forest" of decision trees, where each tree:

1. Is trained on a random subset of the data (bootstrap sampling)
2. Considers only a random subset of features at each split
3. Votes on the final prediction

This randomness makes the trees different from each other, and when combined,
they create a robust classifier that:

- Is resistant to overfitting
- Handles both numerical and categorical features
- Works well with default parameters
- Provides feature importance scores

Example: Predicting customer churn

- Tree 1 might focus on usage patterns and account age
- Tree 2 might focus on customer service calls and billing
- Tree 3 might focus on contract type and payment history
- Together, they give a more reliable prediction than any single tree

## How It Works

Random Forest is a meta estimator that fits a number of decision tree classifiers on
various sub-samples of the dataset and uses averaging to improve predictive accuracy
and control overfitting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomForestClassifier(RandomForestClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the RandomForestClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LeafCount` |  |
| `MaxDepth` |  |
| `NodeCount` |  |
| `OobScore_` | Out-of-bag accuracy score (only available if OobScore is enabled). |
| `Options` | Gets the Random Forest specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateMaxDepth` | Calculates the maximum depth across all trees. |
| `CalculateMaxFeatures` | Calculates the number of features to consider at each split. |
| `CalculateOobScore(Matrix<>,Vector<>)` | Calculates the out-of-bag score. |
| `CalculateTotalLeafCount` | Calculates the total number of leaf nodes across all trees. |
| `CalculateTotalNodeCount` | Calculates the total number of nodes across all trees. |
| `Clone` |  |
| `CreateBootstrapData(Matrix<>,Vector<>,List<Int32>)` | Creates bootstrap sample data matrices. |
| `CreateBootstrapSample(Int32)` | Creates a bootstrap sample of indices. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_oobIndicesPerTree` | Out-of-bag sample indices for each tree. |
| `_random` | Random number generator for bootstrap sampling and feature selection. |

