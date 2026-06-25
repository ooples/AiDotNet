---
title: "KNeighborsOptions<T>"
description: "Configuration options for K-Nearest Neighbors classifiers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for K-Nearest Neighbors classifiers.

## For Beginners

KNN is like asking your neighbors for advice!

When you need to classify a new sample:

1. Find the k training samples closest to it
2. Look at what classes those neighbors belong to
3. Predict the most common class among those neighbors

Example: To predict if a movie is "Action" or "Comedy":

- Find 5 similar movies (based on runtime, budget, etc.)
- If 4 are Action and 1 is Comedy, predict "Action"

Key settings:

- K (N_Neighbors): How many neighbors to consider (default: 5)
- Metric: How to measure distance (Euclidean, Manhattan, etc.)

## How It Works

K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that classifies
samples based on the majority class among their k nearest neighbors in the feature space.

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets or sets the algorithm used to compute nearest neighbors. |
| `LeafSize` | Gets or sets the leaf size for tree-based algorithms. |
| `Metric` | Gets or sets the distance metric used to find nearest neighbors. |
| `NNeighbors` | Gets or sets the number of neighbors to use for classification. |
| `P` | Gets or sets the power parameter for the Minkowski metric. |
| `Weights` | Gets or sets the weight function used in prediction. |

