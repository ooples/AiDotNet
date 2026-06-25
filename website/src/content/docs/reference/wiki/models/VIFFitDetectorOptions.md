---
title: "VIFFitDetectorOptions"
description: "Configuration options for detecting multicollinearity in regression models using Variance Inflation Factor (VIF) analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for detecting multicollinearity in regression models using Variance Inflation Factor (VIF) analysis.

## For Beginners

This class helps you detect when predictor variables in your model are too closely related to each other.

When building regression models:

- Multicollinearity occurs when predictor variables are highly correlated with each other
- This can make your model unstable and difficult to interpret
- Coefficients may change dramatically with small changes in the data
- Standard errors of coefficients become inflated

VIF (Variance Inflation Factor):

- Measures how much the variance of a coefficient is increased due to multicollinearity
- Higher VIF values indicate more severe multicollinearity
- VIF = 1 means no multicollinearity
- VIF > 1 indicates some degree of multicollinearity

This class provides thresholds to automatically detect problematic levels of
multicollinearity in your models, helping you identify when you should consider
removing or combining variables.

## How It Works

Variance Inflation Factor (VIF) is a statistical measure used to detect the severity of multicollinearity in 
regression analysis. Multicollinearity occurs when independent variables in a regression model are highly 
correlated with each other, which can lead to unstable and unreliable coefficient estimates. VIF quantifies 
how much the variance of an estimated regression coefficient is increased due to collinearity with other 
predictors. This class provides configuration options for thresholds used to interpret VIF values and detect 
problematic levels of multicollinearity in regression models. These thresholds help automate the process of 
model evaluation and variable selection.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for determining a good fit in terms of the primary metric. |
| `ModerateMulticollinearityThreshold` | Gets or sets the threshold for detecting moderate multicollinearity. |
| `PrimaryMetric` | Gets or sets the primary metric used to evaluate model fit. |
| `SevereMulticollinearityThreshold` | Gets or sets the threshold for detecting severe multicollinearity. |

