---
title: "ModelStats<T, TInput, TOutput>"
description: "Represents a collection of statistical metrics for evaluating and analyzing machine learning models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Statistics`

Represents a collection of statistical metrics for evaluating and analyzing machine learning models.

## For Beginners

Think of ModelStats as a report card for your AI model.

Just like a school report card shows how well a student is doing in different subjects,
ModelStats shows how well your AI model is performing in different areas. It helps you:

- Understand how accurate your model's predictions are
- See which features (inputs) are most important
- Check if your model is too simple or too complex
- Compare your model's performance to simpler alternatives

This information helps you improve your model and decide if it's ready to use in real-world situations.

## How It Works

This class calculates and stores various statistical measures that help assess the performance,
fit, and characteristics of a machine learning model. It includes metrics for model accuracy,
feature importance, model complexity, and various distance and similarity measures.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelStats(ModelStatsInputs<,,>,ModelStatsOptions)` | Initializes a new instance of the `ModelStats<T>` class with the specified inputs and options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Actual` | Gets the actual (observed) values from the dataset. |
| `AutoCorrelationFunction` | Gets the Auto-Correlation Function, which measures the correlation between a time series and a lagged version of itself. |
| `FeatureCount` | Gets the number of features (input variables) used in the model. |
| `FeatureNames` | Gets the names of the features used in the model. |
| `FeatureValues` | Gets a dictionary mapping feature names to their values. |
| `Features` | Gets the feature values used in the model. |
| `Model` | Gets the full model being evaluated. |
| `PartialAutoCorrelationFunction` | Gets the Partial Auto-Correlation Function, which measures the direct relationship between an observation and its lag. |
| `Predicted` | Gets the predicted values from the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateModelStats(ModelStatsInputs<,,>)` | Calculates all the statistical measures for the model. |
| `Empty` | Creates an empty instance of the `ModelStats<T>` class. |
| `GetMetric(MetricType)` | Retrieves the value of a specific metric. |
| `HasMetric(MetricType)` | Checks if a specific metric is available in this ModelStats instance. |
| `IsEmptyInput(ModelStatsInputs<,,>)` | Determines whether the input data is empty or uninitialized. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_calinskiHarabaszIndex` | Gets the Calinski-Harabasz index, a measure of cluster separation. |
| `_conditionNumber` | Gets the condition number, a measure of the model's numerical stability. |
| `_correlationMatrix` | Gets the correlation matrix showing relationships between features. |
| `_cosineSimilarity` | Gets the cosine similarity between actual and predicted values. |
| `_covarianceMatrix` | Gets the covariance matrix showing how features vary together. |
| `_daviesBouldinIndex` | Gets the Davies-Bouldin index, a measure of the average similarity between each cluster and its most similar cluster. |
| `_effectiveNumberOfParameters` | Gets the effective number of parameters in the model. |
| `_euclideanDistance` | Gets the Euclidean distance between actual and predicted values. |
| `_hammingDistance` | Gets the Hamming distance between actual and predicted values. |
| `_jaccardSimilarity` | Gets the Jaccard similarity between actual and predicted values. |
| `_leaveOneOutPredictiveDensities` | Gets the leave-one-out predictive densities for each data point. |
| `_logLikelihood` | Gets the log-likelihood of the model. |
| `_logPointwisePredictiveDensity` | Gets the log pointwise predictive density, a measure of prediction accuracy. |
| `_mahalanobisDistance` | Gets the Mahalanobis distance between actual and predicted values. |
| `_manhattanDistance` | Gets the Manhattan distance between actual and predicted values. |
| `_marginalLikelihood` | Gets the marginal likelihood of the model. |
| `_meanAveragePrecision` | Gets the Mean Average Precision, a measure of ranking quality. |
| `_meanReciprocalRank` | Gets the Mean Reciprocal Rank, a statistic measuring the performance of a system that produces a list of possible responses to a query. |
| `_mutualInformation` | Gets the mutual information between actual and predicted values. |
| `_normalizedDiscountedCumulativeGain` | Gets the Normalized Discounted Cumulative Gain, a measure of ranking quality that takes the position of correct items into account. |
| `_normalizedMutualInformation` | Gets the normalized mutual information between actual and predicted values. |
| `_observedTestStatistic` | Gets the observed test statistic for model evaluation. |
| `_posteriorPredictiveSamples` | Gets samples from the posterior predictive distribution. |
| `_referenceModelMarginalLikelihood` | Gets the marginal likelihood of a reference (simpler) model. |
| `_silhouetteScore` | Gets the silhouette score, a measure of how similar an object is to its own cluster compared to other clusters. |
| `_vIFList` | Gets the Variance Inflation Factor (VIF) for each feature. |
| `_variationOfInformation` | Gets the variation of information between actual and predicted values. |

