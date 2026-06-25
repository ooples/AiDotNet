---
title: "LossPropertyAttribute"
description: "Declares mathematical properties of a loss function for automatic test generation and cataloging."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Declares mathematical properties of a loss function for automatic test generation and cataloging.

## Properties

| Property | Summary |
|:-----|:--------|
| `ApiShape` | The method signature shape this loss uses for its primary calculation. |
| `ExpectedOutput` | Expected output format. |
| `HandlesImbalancedData` | Whether designed for imbalanced data (Focal, Dice). |
| `HasStandardGradientSign` | Whether the gradient sign follows standard convention (positive when predicted > actual). |
| `IsNonNegative` | Whether the loss is always ≥ 0. |
| `IsRobustToOutliers` | Whether robust to outliers (Huber, MAE). |
| `IsSymmetric` | Whether L(x,y) == L(y,x). |
| `RequiresProbabilityInputs` | Whether inputs must be in [0,1]. |
| `SupportsClassWeights` | Whether per-class weights are supported. |
| `TestInputFormat` | The format of test data that this loss function expects. |
| `ZeroDerivativeForIdentical` | Whether dL/dp == 0 when predicted == actual. |
| `ZeroForIdentical` | Whether L(x,x) == 0. |

