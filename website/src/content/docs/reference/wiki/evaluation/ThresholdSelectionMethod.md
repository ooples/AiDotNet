---
title: "ThresholdSelectionMethod"
description: "Specifies the method for selecting classification thresholds."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Enums`

Specifies the method for selecting classification thresholds.

## For Beginners

If your model says "70% chance of fraud", should you flag it as fraud?
That depends on your threshold. At 0.5 threshold, yes (70% > 50%). At 0.8 threshold, no (70% < 80%).
Different methods help you pick the best threshold for your needs.

## How It Works

Classification models often output probabilities (0.0 to 1.0). A threshold determines
when to classify as positive vs negative. The default 0.5 isn't always optimal.

## Fields

| Field | Summary |
|:-----|:--------|
| `ClosestToTopLeft` | Closest to (0,1) on ROC: Point on ROC curve closest to perfect classification. |
| `CostSensitive` | Cost-sensitive: Minimizes expected cost based on false positive/negative costs. |
| `Default` | Default threshold of 0.5. |
| `F1Max` | F1-score maximization: Selects threshold that maximizes F1 score. |
| `FBetaMax` | F-beta maximization: Maximizes F-beta score with configurable beta. |
| `FixedNPV` | Fixed NPV: Choose threshold to achieve a target negative predictive value. |
| `FixedPrecision` | Fixed precision: Choose threshold to achieve a target precision (PPV). |
| `FixedSensitivity` | Fixed sensitivity: Choose threshold to achieve a target sensitivity (recall). |
| `FixedSpecificity` | Fixed specificity: Choose threshold to achieve a target specificity. |
| `GeometricMean` | Geometric mean maximization: Maximizes sqrt(sensitivity * specificity). |
| `Informedness` | Informedness maximization: Maximizes informedness (same as Youden's J). |
| `MCCMax` | Matthews Correlation Coefficient maximization: Maximizes MCC. |
| `Markedness` | Markedness maximization: Maximizes PPV + NPV - 1. |
| `PrecisionRecallBreakeven` | Precision-recall breakeven: Where precision equals recall. |
| `PrevalenceAdjusted` | Prevalence-adjusted: Adjusts threshold based on class prevalence. |
| `Youden` | Youden's J statistic: Maximizes sensitivity + specificity - 1. |

