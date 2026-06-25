---
title: "BootstrapFitDetectorOptions"
description: "Configuration options for the Bootstrap Fit Detector, which evaluates model fit quality using bootstrap resampling."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Bootstrap Fit Detector, which evaluates model fit quality using bootstrap resampling.

## For Beginners

This class contains settings for a tool that helps you determine if your AI model is 
working well. It uses a technique called "bootstrapping" - imagine randomly picking data points from your dataset 
(sometimes picking the same point multiple times) to create many similar-but-different datasets. By training your 
model on these different datasets and seeing how consistent the results are, we can tell if your model is learning 
real patterns or just memorizing the training data.

## How It Works

Bootstrap resampling is a statistical technique that creates multiple datasets by randomly sampling with replacement
from the original dataset. By training models on these resampled datasets and evaluating their performance,
we can assess how well a model generalizes and detect issues like overfitting or underfitting.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceInterval` | Gets or sets the confidence interval used for statistical assessments. |
| `GoodFitThreshold` | Gets or sets the threshold for identifying a good model fit. |
| `NumberOfBootstraps` | Gets or sets the number of bootstrap samples to generate for evaluation. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting. |

