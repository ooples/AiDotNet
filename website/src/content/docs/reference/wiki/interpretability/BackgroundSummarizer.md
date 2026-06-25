---
title: "BackgroundSummarizer<T>"
description: "Provides methods for summarizing background data for SHAP and other interpretability methods."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Interpretability.Helpers`

Provides methods for summarizing background data for SHAP and other interpretability methods.

## For Beginners

SHAP and similar methods need "background data" to compare against.
But using all your training data as background can be VERY slow.

This class provides ways to summarize large datasets into smaller representative sets:

- **KMeans:** Group similar samples together, use cluster centers as representatives
- **Stratified Sampling:** Sample proportionally from different groups/categories
- **Random Sampling:** Simple random subset selection

**Example:** If you have 10,000 training samples, you might summarize to 100 samples.
SHAP computation goes from O(10000) to O(100) = 100x faster!

**Trade-off:**

- Fewer background samples = faster but less accurate
- More background samples = slower but more accurate
- 50-200 samples is usually a good balance

## Methods

| Method | Summary |
|:-----|:--------|
| `Auto(Matrix<>,Int32,Int32[],Nullable<Int32>)` | Automatically chooses the best summarization method. |
| `ComputeDistance(Vector<>,Vector<>)` | Computes Euclidean distance between two vectors. |
| `CreateUniformWeights(Int32)` | Creates uniform weights that sum to 1. |
| `FindNearestCenter(Vector<>,Matrix<>)` | Finds the nearest center for a data point. |
| `InitializeCentersKMeansPlusPlus(Matrix<>,Int32,Random)` | Initializes cluster centers using KMeans++ algorithm. |
| `KMeans(Matrix<>,Int32,Int32,Nullable<Int32>)` | Summarizes data using KMeans clustering. |
| `MixedSummary(Matrix<>,Int32[],Int32,Nullable<Int32>)` | Summarizes data using a combination of methods for mixed data. |
| `RandomSample(Matrix<>,Int32,Nullable<Int32>)` | Summarizes data using random sampling. |
| `StratifiedSample(Matrix<>,Int32,Int32,Nullable<Int32>)` | Summarizes data using stratified sampling. |

