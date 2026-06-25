---
title: "AnomalyFeatures"
description: "Flags for selecting which anomaly detection features to calculate."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Flags for selecting which anomaly detection features to calculate.

## Fields

| Field | Summary |
|:-----|:--------|
| `All` | All anomaly detection features. |
| `ControlChartFeatures` | Control chart features (CUSUM). |
| `CusumFlag` | CUSUM change point flag: 1 when CUSUM exceeds threshold, 0 otherwise. |
| `CusumStatistic` | CUSUM (Cumulative Sum) statistic for detecting mean shifts. |
| `IqrFeatures` | IQR-based anomaly detection features. |
| `IqrOutlierFlag` | IQR outlier flag: 1 if value is outside IQR bounds, 0 otherwise. |
| `IqrOutlierScore` | IQR outlier score: distance from the nearest quartile boundary. |
| `IsolationScore` | Isolation score: higher values indicate more anomalous points. |
| `ModifiedZScore` | Modified Z-score using median absolute deviation (more robust to outliers). |
| `None` | No anomaly features. |
| `PercentileRank` | Percentile rank: where the current value falls in the rolling distribution (0-1). |
| `ZScore` | Rolling Z-score: measures how many standard deviations a value is from the rolling mean. |
| `ZScoreFeatures` | Z-score based anomaly detection features. |
| `ZScoreFlag` | Z-score anomaly flag: 1 if \|Z-score\| exceeds threshold, 0 otherwise. |

