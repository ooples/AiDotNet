---
title: "ValidationCurveResult<T>"
description: "Results from validation curve analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Results from validation curve analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `CVFolds` | Number of CV folds used at each parameter value. |
| `MetricName` | Name of the metric tracked. |
| `OptimalParameterValue` | Optimal parameter value based on validation score. |
| `ParameterName` | Name of the hyperparameter varied. |
| `ParameterValues` | Hyperparameter values tested. |
| `TrainScoreMeans` | Mean training scores at each parameter value. |
| `TrainScoreStds` | Standard deviation of training scores. |
| `ValidationScoreMeans` | Mean validation scores at each parameter value. |
| `ValidationScoreStds` | Standard deviation of validation scores. |

