---
title: "ResidualBootstrapFitDetectorOptions"
description: "Configuration options for the Residual Bootstrap Fit Detector, which uses bootstrap resampling of residuals to assess model fit quality and detect overfitting or underfitting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models`

Configuration options for the Residual Bootstrap Fit Detector, which uses bootstrap resampling
of residuals to assess model fit quality and detect overfitting or underfitting.

## For Beginners

This class helps detect if your model is learning too much or too little from your data.

Two common problems in machine learning are:

- Overfitting: When your model learns the training data too well, including random noise
- Underfitting: When your model is too simple and misses important patterns in the data

This detector uses a technique called "bootstrap resampling" to check for these problems:

- It creates many random samples from your original data (with replacement)
- For each sample, it calculates how well the model performs
- By comparing performance across these samples, it can detect overfitting or underfitting

Think of it like testing a student:

- Overfitting is like memorizing the textbook but failing to understand the concepts
- Underfitting is like not studying enough and missing basic information
- This detector helps identify which problem your model might have

The settings in this class control how thoroughly and strictly this testing process works.

## How It Works

Bootstrap resampling is a statistical technique that involves repeatedly sampling with replacement 
from the original dataset to estimate the sampling distribution of a statistic. The Residual Bootstrap 
Fit Detector applies this technique to model residuals (the differences between predicted and actual values) 
to assess whether a model is overfitting or underfitting the data. Overfitting occurs when a model learns 
the training data too well, including its noise, resulting in poor generalization to new data. Underfitting 
occurs when a model is too simple to capture the underlying patterns in the data. This class provides 
configuration options for the bootstrap process, including the number of bootstrap samples to generate, 
minimum sample size requirements, thresholds for detecting overfitting and underfitting, and an optional 
seed for reproducibility.

## Properties

| Property | Summary |
|:-----|:--------|
| `MinSampleSize` | Gets or sets the minimum sample size required for bootstrap analysis. |
| `NumBootstrapSamples` | Gets or sets the number of bootstrap samples to generate for the analysis. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting in the model. |
| `Seed` | Gets or sets the random seed for the bootstrap sampling process. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting in the model. |

