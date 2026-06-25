---
title: "ErrorStats<T>"
description: "Calculates and stores various error metrics for evaluating prediction model performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Statistics`

Calculates and stores various error metrics for evaluating prediction model performance.

## For Beginners

When building AI or machine learning models, you need ways to measure how accurate your predictions are.
Think of these metrics like different ways to score a test:

## How It Works

This class provides a comprehensive set of error metrics to assess how well predicted values 
match actual values.

The "T" in ErrorStats<T> means this class works with different number types like decimal,
double, or float without needing separate implementations for each.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ErrorStats(ErrorStatsInputs<>)` | Creates a new ErrorStats instance and calculates all error metrics. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AUC` | Area Under the Curve (ROC) - Alias for AUCROC property. |
| `MeanAbsoluteError` | Mean Absolute Error - Alias for MAE property. |
| `MeanSquaredError` | Mean Squared Error - Alias for MSE property. |
| `RootMeanSquaredError` | Root Mean Squared Error - Alias for RMSE property. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateErrorStats(Vector<>,Vector<>,Int32,PredictionType)` | Calculates all error metrics based on actual and predicted values. |
| `Empty` | Creates an empty ErrorStats instance with all metrics set to zero. |
| `GetMetric(MetricType)` | Retrieves the value of a specific error metric. |
| `HasMetric(MetricType)` | Checks if a specific metric is available in this ErrorStats instance. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_aIC` | Akaike Information Criterion - A measure that balances model accuracy and complexity. |
| `_aICAlt` | Alternative Akaike Information Criterion - A variant of AIC with a different penalty term. |
| `_aUCPR` | Area Under the Precision-Recall Curve - Measures classification accuracy focusing on positive cases. |
| `_aUCROC` | Area Under the Receiver Operating Characteristic Curve - Measures classification accuracy across thresholds. |
| `_accuracy` | Classification accuracy - The proportion of correct predictions (for classification tasks). |
| `_bIC` | Bayesian Information Criterion - Similar to AIC but penalizes model complexity more strongly. |
| `_cRPS` | Continuous Ranked Probability Score - Evaluates probabilistic forecast accuracy. |
| `_durbinWatsonStatistic` | Durbin-Watson Statistic - Detects autocorrelation in prediction errors. |
| `_errorList` | List of individual prediction errors (residuals). |
| `_f1Score` | The harmonic mean of precision and recall (for classification). |
| `_mAE` | Mean Absolute Error - The average absolute difference between predicted and actual values. |
| `_mAPE` | Mean Absolute Percentage Error - The average percentage difference between predicted and actual values. |
| `_mSE` | Mean Squared Error - The average of squared differences between predicted and actual values. |
| `_maxError` | Maximum Error - The largest absolute difference between any predicted and actual value. |
| `_meanBiasError` | Mean Bias Error - The average of prediction errors (predicted - actual). |
| `_meanSquaredLogError` | Mean Squared Logarithmic Error - Penalizes underestimates more than overestimates. |
| `_medianAbsoluteError` | Median Absolute Error - The middle value of all absolute differences between predicted and actual values. |
| `_numOps` | Provides mathematical operations for the generic type T. |
| `_populationStandardError` | Population Standard Error - The standard deviation of prediction errors without adjustment for model complexity. |
| `_precision` | The proportion of positive predictions that were actually correct (for classification). |
| `_rMSE` | Root Mean Squared Error - The square root of the Mean Squared Error. |
| `_rSS` | Residual Sum of Squares - The sum of squared differences between predicted and actual values. |
| `_recall` | The proportion of actual positive cases that were correctly identified (for classification). |
| `_sMAPE` | Symmetric Mean Absolute Percentage Error - A variant of MAPE that handles zero or near-zero values better. |
| `_sampleStandardError` | Sample Standard Error - An estimate of the standard deviation of prediction errors, adjusted for model complexity. |
| `_theilUStatistic` | Theil's U Statistic - A measure of forecast accuracy relative to a naive forecasting method. |

