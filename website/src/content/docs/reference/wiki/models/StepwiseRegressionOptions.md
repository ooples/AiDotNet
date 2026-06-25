---
title: "StepwiseRegressionOptions<T>"
description: "Configuration options for Stepwise Regression, an automated feature selection approach that iteratively adds or removes predictors based on their statistical significance."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Stepwise Regression, an automated feature selection approach
that iteratively adds or removes predictors based on their statistical significance.

## For Beginners

Stepwise Regression helps automatically select the most important variables for your model.

When building a regression model:

- You often have many potential predictor variables
- Not all variables are equally useful
- Including too many variables can lead to overfitting
- Including too few might miss important relationships

Stepwise regression solves this by:

- Systematically testing different combinations of variables
- Adding or removing variables one at a time
- Keeping only those that significantly improve the model
- Stopping when further changes don't help much

This approach helps you:

- Identify which variables actually matter
- Create simpler, more interpretable models
- Avoid the computational cost of unnecessary variables
- Potentially improve prediction accuracy

This class lets you configure exactly how the stepwise selection process works.

## How It Works

Stepwise Regression is an automated approach to building regression models by iteratively adding or removing 
predictor variables based on their statistical significance. This technique helps identify the most important 
features while excluding those that contribute little to the model's predictive power, resulting in more 
parsimonious and potentially more interpretable models. There are several variants of stepwise regression, 
including forward selection (starting with no predictors and adding them one by one), backward elimination 
(starting with all predictors and removing them one by one), and bidirectional elimination (a combination of 
both approaches). This class provides configuration options for controlling the stepwise regression process, 
including the selection method, constraints on the number of features, and criteria for determining when to 
stop adding or removing features.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxFeatures` | Gets or sets the maximum number of features to include in the final model. |
| `Method` | Gets or sets the stepwise selection method to use. |
| `MinFeatures` | Gets or sets the minimum number of features to include in the final model. |
| `MinImprovement` | Gets or sets the minimum improvement in the model's fit statistic required to add or remove a feature. |

