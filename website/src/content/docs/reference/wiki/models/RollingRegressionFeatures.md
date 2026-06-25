---
title: "RollingRegressionFeatures"
description: "Flags for selecting which rolling regression features to calculate."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Flags for selecting which rolling regression features to calculate.

## Fields

| Field | Summary |
|:-----|:--------|
| `All` | All regression features. |
| `Alpha` | Rolling Alpha - excess return over benchmark. |
| `Beta` | Rolling Beta - measures asset's sensitivity to benchmark movements. |
| `CAPMFeatures` | Standard CAPM regression features (Alpha, Beta, R²). |
| `Correlation` | Rolling Correlation with benchmark. |
| `InformationRatio` | Rolling Information Ratio - alpha per unit of tracking error. |
| `None` | No regression features. |
| `RSquared` | Rolling R-squared - coefficient of determination. |
| `RiskAdjusted` | Risk-adjusted return measures. |
| `SharpeRatio` | Rolling Sharpe Ratio - risk-adjusted return measure. |
| `SortinoRatio` | Rolling Sortino Ratio - downside risk-adjusted return measure. |
| `TrackingError` | Rolling Tracking Error - standard deviation of return differences. |

