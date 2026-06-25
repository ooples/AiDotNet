---
title: "DatasetStatistics<T>"
description: "Statistics about the dataset used for evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Results.Core`

Statistics about the dataset used for evaluation.

## For Beginners

Before trusting evaluation metrics, you need to understand your data.
This class tells you:

- How many samples were evaluated
- Class distribution (for classification) - are classes balanced?
- Target variable statistics (for regression) - range and spread
- Missing values, outliers, and other data quality issues

## How It Works

Contains summary information about the evaluation dataset including sample counts,
class distributions, feature statistics, and data quality indicators.

## Properties

| Property | Summary |
|:-----|:--------|
| `AppearsTimeSeries` | Whether the dataset appears to be a time series (ordered, potentially autocorrelated). |
| `ClassCounts` | Number of samples per class. |
| `ClassImbalanceRatio` | Class imbalance ratio (largest class / smallest class). |
| `ClassLabels` | Class labels (for classification). |
| `ClassProportions` | Proportion of samples per class. |
| `DuplicateCount` | Number of duplicate samples. |
| `FeatureNames` | Feature names, if available. |
| `FeatureStats` | Per-feature statistics (min, max, mean, std, missing count). |
| `IsClassification` | Whether this is a classification task. |
| `IsImbalanced` | Whether classes are considered imbalanced (ratio > threshold). |
| `IsMultiLabel` | Whether this is a multi-label classification task. |
| `MissingProportion` | Proportion of samples with missing values. |
| `MissingSamplesCount` | Number of samples with missing values. |
| `NumberOfClasses` | Number of unique classes (for classification). |
| `NumberOfFeatures` | Number of features in the dataset. |
| `NumberOfOutputs` | Number of output dimensions (1 for scalar, >1 for multi-output). |
| `OutlierCount` | Number of detected outliers. |
| `OutlierProportion` | Proportion of outliers. |
| `TargetKurtosis` | Target value kurtosis (for regression). |
| `TargetMax` | Maximum target value (for regression). |
| `TargetMean` | Mean target value (for regression). |
| `TargetMedian` | Median target value (for regression). |
| `TargetMin` | Minimum target value (for regression). |
| `TargetSkewness` | Target value skewness (for regression). |
| `TargetStdDev` | Standard deviation of target (for regression). |
| `TotalSamples` | Total number of samples in the evaluation set. |
| `Warnings` | Warnings about the dataset (imbalance, outliers, etc.). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Gets a summary string describing the dataset. |

