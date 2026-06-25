---
title: "ResidualAnalysisOptions"
description: "Configuration options for residual analysis in regression models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for residual analysis in regression models.

## For Beginners

Residuals are errors: Actual - Predicted. Good residuals should:

- Have mean zero (no systematic bias)
- Be normally distributed (for inference)
- Have constant variance (homoscedasticity)
- Be independent (no patterns over time or by predicted value)

If these assumptions are violated, your model might be missing something.

## How It Works

Residual analysis examines the differences between predicted and actual values to
diagnose model problems like heteroscedasticity, non-normality, and autocorrelation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComputeDeletedResiduals` | Whether to compute deleted residuals. |
| `ComputePartialResidualPlots` | Whether to compute partial residual plots. |
| `ComputeResidualACF` | Whether to compute residual ACF/PACF for autocorrelation. |
| `ComputeStandardizedResiduals` | Whether to compute standardized residuals. |
| `ComputeStudentizedResiduals` | Whether to compute studentized residuals. |
| `GenerateQQPlot` | Whether to generate Q-Q plot data. |
| `GenerateResidualLeveragePlot` | Whether to generate residual vs leverage plot data. |
| `GenerateResidualVsFittedPlot` | Whether to generate residual vs fitted plot data. |
| `GenerateScaleLocationPlot` | Whether to generate scale-location plot data. |
| `IdentifyOutliers` | Whether to identify outliers. |
| `LjungBoxLag` | Lag for Ljung-Box test. |
| `MaxACFLag` | Maximum lag for ACF computation. |
| `MaxOutliersToReport` | Maximum number of outliers to report. |
| `OutlierThreshold` | Threshold for outlier detection (standard deviations). |
| `PartialResidualFeatures` | Features for partial residual plots. |
| `RESETPowers` | Power terms for RESET test. |
| `RunBreuschPaganTest` | Whether to run Breusch-Pagan test for heteroscedasticity. |
| `RunDurbinWatsonTest` | Whether to run Durbin-Watson test for autocorrelation. |
| `RunGoldfeldQuandtTest` | Whether to run Goldfeld-Quandt test. |
| `RunJarqueBeraTest` | Whether to run Jarque-Bera normality test. |
| `RunKolmogorovSmirnovTest` | Whether to run Kolmogorov-Smirnov test. |
| `RunLjungBoxTest` | Whether to run Ljung-Box test for autocorrelation. |
| `RunRESETTest` | Whether to run RESET test for functional form. |
| `RunShapiroWilkTest` | Whether to run Shapiro-Wilk normality test. |
| `RunWhiteTest` | Whether to run White test for heteroscedasticity. |
| `SignificanceLevel` | Significance level for all tests. |

