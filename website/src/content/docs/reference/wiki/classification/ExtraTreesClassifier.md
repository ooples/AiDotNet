---
title: "ExtraTreesClassifier<T>"
description: "Extra Trees (Extremely Randomized Trees) classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Ensemble`

Extra Trees (Extremely Randomized Trees) classifier.

## For Beginners

Extra Trees takes randomization even further than Random Forest:

Random Forest: "Look at random features, pick the BEST split"
Extra Trees: "Look at random features, pick a RANDOM split"

Benefits of Extra Trees:

- Faster training (no need to find optimal splits)
- Often better generalization
- More robust to noise

When Extra Trees might be better:

- When you have noisy data
- When Random Forest overfits
- When you need faster training

## How It Works

Extra Trees is an ensemble method that builds multiple decision trees with
extra randomization. Unlike Random Forest which finds the best split among
random features, Extra Trees picks random splits, leading to more diversity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExtraTreesClassifier(ExtraTreesClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the ExtraTreesClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LeafCount` |  |
| `MaxDepth` |  |
| `NodeCount` |  |
| `Options` | Gets the Extra Trees specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateMaxDepth` | Calculates the maximum depth across all trees. |
| `CalculateMaxFeatures` | Calculates the number of features to consider at each split. |
| `CalculateTotalLeafCount` | Calculates the total number of leaf nodes. |
| `CalculateTotalNodeCount` | Calculates the total number of nodes. |
| `Clone` |  |
| `CreateBootstrapSample(Matrix<>,Vector<>)` | Creates a bootstrap sample. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_random` | Random number generator. |

