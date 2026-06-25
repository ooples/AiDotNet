---
title: "PredictionStats<T>"
description: "Calculates and stores various statistics to evaluate prediction performance and generate prediction intervals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Statistics`

Calculates and stores various statistics to evaluate prediction performance and generate prediction intervals.

## How It Works

This class provides a comprehensive set of metrics to evaluate predictive models and 
calculate different types of statistical intervals around predictions.

For Beginners:
When you build a predictive model (like a machine learning model), you often want to:

1. Measure how well your model performs (using metrics like R-squared (R2), accuracy, etc.)
2. Understand how confident you can be in your predictions (using various intervals)
3. Understand the relationship between actual and predicted values (using correlations)

This class helps you do all of these things. The "T" in PredictionStats<T> means it 
works with different number types like decimal, double, or float without needing separate 
implementations for each.

## Properties

| Property | Summary |
|:-----|:--------|
| `R2` | Coefficient of determination - The proportion of variance in the dependent variable explained by the model. |
| `RSquared` | R-Squared - Alias for R2 property (Coefficient of determination). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePredictionStats(Vector<>,Vector<>,Int32,,Int32,PredictionType)` | Calculates all prediction statistics based on actual and predicted values. |
| `Empty` | Creates an empty PredictionStats instance with all metrics set to zero. |
| `EnsureFullStatsComputed` | Ensures all stats beyond R²/AdjustedR² are computed. |
| `GetMetric(MetricType)` | Retrieves a specific metric value by metric type. |
| `HasMetric(MetricType)` | Checks if a specific metric is available in this PredictionStats instance. |
| `WithR2Only()` | Creates a lightweight PredictionStats with only the R² coefficient populated. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_accuracy` | The proportion of predictions that the model got correct (for classification). |
| `_bestDistributionFit` | Information about the statistical distribution that best fits the prediction data. |
| `_bootstrapInterval` | Bootstrap Interval - An interval created using resampling techniques. |
| `_confidenceInterval` | Confidence Interval - A range that likely contains the true mean of the predictions. |
| `_credibleInterval` | Credible Interval - A Bayesian interval that contains the true value with a certain probability. |
| `_deferredInputs` | Creates a new PredictionStats instance and calculates all prediction metrics. |
| `_dynamicTimeWarping` | A measure of similarity between two temporal sequences. |
| `_explainedVarianceScore` | The explained variance score - A measure of how well the model accounts for the variance in the data. |
| `_f1Score` | The harmonic mean of precision and recall (for classification). |
| `_forecastInterval` | Forecast Interval - A prediction interval specifically for time series forecasting. |
| `_jackknifeInterval` | Jackknife Interval - An interval created by systematically leaving out one observation at a time. |
| `_kendallTau` | A measure of concordance between actual and predicted values based on paired rankings. |
| `_learningCurve` | A list of performance metrics calculated at different training set sizes. |
| `_meanPredictionError` | The average of all prediction errors (predicted - actual). |
| `_medianPredictionError` | The middle value of all prediction errors (predicted - actual). |
| `_numOps` | Provides mathematical operations for the generic type T. |
| `_pearsonCorrelation` | A measure of linear correlation between actual and predicted values. |
| `_percentileInterval` | Percentile Interval - An interval based directly on the percentiles of the prediction distribution. |
| `_precision` | The proportion of positive predictions that were actually correct (for classification). |
| `_predictionInterval` | Prediction Interval - A range that likely contains future individual observations. |
| `_predictionIntervalCoverage` | The proportion of actual values that fall within the prediction interval. |
| `_quantileIntervals` | Collection of prediction intervals at different quantile levels. |
| `_recall` | The proportion of actual positive cases that were correctly identified (for classification). |
| `_simultaneousPredictionInterval` | Simultaneous Prediction Interval - A prediction interval that accounts for multiple predictions. |
| `_spearmanCorrelation` | A measure of monotonic correlation between the ranks of actual and predicted values. |
| `_toleranceInterval` | Tolerance Interval - A range that contains a specified proportion of the population with a certain confidence. |

