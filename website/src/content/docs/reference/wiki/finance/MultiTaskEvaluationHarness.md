---
title: "MultiTaskEvaluationHarness<T>"
description: "Multi-task evaluation harness for time series foundation models, supporting standardized evaluation across forecasting, anomaly detection, classification, imputation, and embedding tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Evaluation`

Multi-task evaluation harness for time series foundation models, supporting standardized
evaluation across forecasting, anomaly detection, classification, imputation, and embedding tasks.

## For Beginners

Foundation models like MOMENT support multiple tasks with one model.
This harness lets you evaluate how well a model performs across all its supported tasks
using standardized metrics for each:

- **Forecasting:** MSE, MAE, RMSE, MASE
- **Anomaly Detection:** Precision, Recall, F1-Score
- **Classification:** Accuracy, F1-Score
- **Imputation:** MSE on imputed values
- **Embedding:** Silhouette score for clustering quality

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAnomalyDetection(ITimeSeriesFoundationModel<>,IReadOnlyList<Tensor<>>,IReadOnlyList<Tensor<>>,Nullable<Double>)` | Evaluates a foundation model's anomaly detection performance. |
| `EvaluateClassification(ITimeSeriesFoundationModel<>,IReadOnlyList<Tensor<>>,IReadOnlyList<Int32>,Int32)` | Evaluates a foundation model's classification performance. |
| `EvaluateForecasting(ITimeSeriesFoundationModel<>,IReadOnlyList<ValueTuple<Tensor<>,Tensor<>>>)` | Evaluates a foundation model's forecasting performance. |
| `EvaluateImputation(ITimeSeriesFoundationModel<>,IReadOnlyList<Tensor<>>,IReadOnlyList<Tensor<>>)` | Evaluates a foundation model's imputation performance. |
| `RunFullEvaluation(ITimeSeriesFoundationModel<>,IReadOnlyList<ValueTuple<Tensor<>,Tensor<>>>,Nullable<ValueTuple<IReadOnlyList<Tensor<>>,IReadOnlyList<Tensor<>>,Nullable<Double>>>,Nullable<ValueTuple<IReadOnlyList<Tensor<>>,IReadOnlyList<Int32>,Int32>>,Nullable<ValueTuple<IReadOnlyList<Tensor<>>,IReadOnlyList<Tensor<>>>>)` | Runs a full multi-task evaluation across all supported tasks. |

