---
title: "ShapleyValueFitDetectorOptions"
description: "Configuration options for the Shapley Value Fit Detector, which evaluates model fit quality by analyzing feature importance using Shapley values."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models`

Configuration options for the Shapley Value Fit Detector, which evaluates model fit quality
by analyzing feature importance using Shapley values.

## For Beginners

This class helps evaluate if your model is using features appropriately.

Shapley values measure how much each feature contributes to your model's predictions:

- They come from game theory and provide a fair way to distribute "credit" among features
- They show which features are doing the heavy lifting in your model
- They can reveal if your model is using features in a balanced way

The detector uses these values to identify potential problems:

- Overfitting: When your model relies too heavily on just a few features (like memorizing the data)
- Underfitting: When your model spreads importance too evenly across many features (not finding strong patterns)

For example, in a house price prediction model:

- Overfitting might show up as the model relying almost entirely on exact address
- Underfitting might show up as the model giving similar importance to crucial factors (like location)

and irrelevant ones (like the day of week the house was listed)

This class lets you configure how the detector evaluates feature importance distribution
to identify these potential issues.

## How It Works

Shapley values, derived from cooperative game theory, provide a method for fairly distributing the 
"contribution" of each feature to the prediction made by a machine learning model. The Shapley Value 
Fit Detector uses these values to assess model fit by analyzing the distribution of feature importance. 
It identifies which features contribute significantly to the model's predictions and uses this information 
to detect potential overfitting (when the model relies too heavily on too few features) or underfitting 
(when the model distributes importance too evenly across many features). This class provides configuration 
options for the thresholds used in this analysis, including the cumulative importance threshold for 
identifying significant features, the number of Monte Carlo samples for calculating Shapley values, 
and thresholds for detecting overfitting and underfitting based on the ratio of important features.

## Properties

| Property | Summary |
|:-----|:--------|
| `ImportanceThreshold` | Gets or sets the threshold for cumulative importance to determine significant features. |
| `MonteCarloSamples` | Gets or sets the number of Monte Carlo samples to use when calculating Shapley values. |
| `OverfitThreshold` | Gets or sets the threshold for the ratio of important features to total features, below which the model is considered to be overfitting. |
| `UnderfitThreshold` | Gets or sets the threshold for the ratio of important features to total features, above which the model is considered to be underfitting. |

