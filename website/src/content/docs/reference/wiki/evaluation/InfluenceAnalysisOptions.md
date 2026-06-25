---
title: "InfluenceAnalysisOptions"
description: "Configuration options for influence analysis in regression models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for influence analysis in regression models.

## For Beginners

Some data points have more influence on the model than others:

- **High leverage:** Point has unusual X values (far from center)
- **High influence:** Removing this point significantly changes the model

A point can be high leverage without being influential (if it fits the pattern).
Influential points deserve extra scrutiny - they might be errors or key insights.

## How It Works

Influence analysis identifies data points that have outsized impact on model predictions
or parameter estimates. These influential points may be outliers, errors, or important cases.

## Properties

| Property | Summary |
|:-----|:--------|
| `AddedVariableFeatures` | Features for added variable plots. |
| `ComputeApproximateLOOImpact` | Whether to compute approximate leave-one-out impact. |
| `ComputeCooksDistance` | Whether to compute Cook's distance. |
| `ComputeCovarianceRatio` | Whether to compute covariance ratio. |
| `ComputeDFBETAS` | Whether to compute DFBETAS. |
| `ComputeDFFITS` | Whether to compute DFFITS. |
| `ComputeExactLOOImpact` | Whether to compute exact leave-one-out (refit for each point). |
| `ComputeHadisMeasure` | Whether to compute Hadi's influence measure. |
| `ComputeLeverage` | Whether to compute leverage (hat values). |
| `ComputeWelschKuhDistance` | Whether to compute Welsch-Kuh distance. |
| `CooksDistanceThreshold` | Cook's distance threshold for flagging. |
| `DFBETASCoefficients` | Specific coefficients to compute DFBETAS for. |
| `DFBETASThreshold` | DFBETAS threshold for flagging. |
| `DFFITSThreshold` | DFFITS threshold for flagging. |
| `ExactLOOSubset` | Subset of points for exact LOO (indices). |
| `GenerateAddedVariablePlots` | Whether to generate added variable plots. |
| `GenerateCPRPlots` | Whether to generate component-plus-residual plots. |
| `GenerateInfluencePlot` | Whether to generate influence plot data. |
| `IdentifyInfluentialPoints` | Whether to identify influential points. |
| `IncludeRecommendations` | Whether to include recommendations for handling influential points. |
| `LeverageThreshold` | Leverage threshold for flagging. |
| `MaxInfluentialPointsToReport` | Maximum number of influential points to report. |
| `PredictionIndicesToAnalyze` | Prediction indices to analyze influence for. |
| `ReportPredictionInfluence` | Whether to report influence on specific predictions. |

