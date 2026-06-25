---
title: "ROCCurveFitDetectorOptions"
description: "Configuration options for the ROC Curve Fit Detector, which evaluates classification model quality using Receiver Operating Characteristic (ROC) curve analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models`

Configuration options for the ROC Curve Fit Detector, which evaluates classification model quality
using Receiver Operating Characteristic (ROC) curve analysis.

## For Beginners

This class helps evaluate how well your classification model performs.

Classification models predict categories (like "spam/not spam" or "will buy/won't buy"):

- We need ways to measure how good these predictions are
- The ROC curve is a powerful tool for this evaluation
- It shows the tradeoff between correctly identifying positives and incorrectly flagging negatives

The key metric is AUC (Area Under the Curve):

- AUC ranges from 0 to 1
- 1.0 means perfect predictions
- 0.5 means the model is no better than random guessing
- Values below 0.5 suggest the model is worse than guessing

This class lets you set thresholds for what AUC values are considered:

- Good performance
- Moderate performance
- Poor performance

It also includes settings to adjust for confidence levels and data imbalance (when you have
many more examples of one category than another).

## How It Works

The ROC Curve Fit Detector assesses classification model performance by analyzing the Receiver Operating 
Characteristic (ROC) curve and its associated Area Under the Curve (AUC) metric. The ROC curve plots the 
True Positive Rate against the False Positive Rate at various classification thresholds, providing a 
comprehensive view of classifier performance across all possible decision thresholds. The AUC value ranges 
from 0 to 1, where 1 represents a perfect classifier and 0.5 represents a classifier that performs no better 
than random guessing. This class provides configuration options for thresholds that determine what constitutes 
good, moderate, and poor model fit based on AUC values, as well as parameters for confidence scaling and 
handling class imbalance. These settings allow users to customize the fit detection criteria according to 
their specific application requirements and domain knowledge.

## Properties

| Property | Summary |
|:-----|:--------|
| `BalancedDatasetThreshold` | Gets or sets the threshold for determining if a dataset is balanced. |
| `ConfidenceScalingFactor` | Gets or sets the scaling factor for confidence intervals when evaluating model fit. |
| `GoodFitThreshold` | Gets or sets the AUC threshold for considering a model to have good fit. |
| `ModerateFitThreshold` | Gets or sets the AUC threshold for considering a model to have moderate fit. |
| `PoorFitThreshold` | Gets or sets the AUC threshold for considering a model to have poor fit. |

