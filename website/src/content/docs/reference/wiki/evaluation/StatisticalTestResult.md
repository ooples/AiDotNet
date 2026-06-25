---
title: "StatisticalTestResult<T>"
description: "Represents the result of a statistical test."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Represents the result of a statistical test.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | The significance level (alpha) used for the test. |
| `ConfidenceInterval` | Confidence interval for the effect, if computed. |
| `DegreesOfFreedom` | Degrees of freedom, if applicable. |
| `Description` | Description of the test and its interpretation. |
| `EffectSize` | Effect size (e.g., Cohen's d), if computed. |
| `Interpretation` | Human-readable interpretation of the test result. |
| `IsSignificant` | Whether the result is statistically significant at the specified alpha level. |
| `PValue` | The p-value (probability of observing this result under null hypothesis). |
| `Statistic` | The test statistic value. |
| `TestName` | Name of the test performed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Formats the result for display. |

