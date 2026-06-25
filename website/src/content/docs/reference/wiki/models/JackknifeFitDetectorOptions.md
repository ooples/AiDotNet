---
title: "JackknifeFitDetectorOptions"
description: "Configuration options for the Jackknife Fit Detector, which uses the jackknife resampling technique to evaluate model stability and detect overfitting or underfitting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Jackknife Fit Detector, which uses the jackknife resampling technique
to evaluate model stability and detect overfitting or underfitting.

## For Beginners

The Jackknife Fit Detector helps you understand if your model is reliable
by testing how it performs when small parts of your data are removed.

Imagine you're building a recipe and want to know if it's robust. You might try leaving out one
ingredient at a time to see how much the taste changes. If leaving out any single ingredient drastically
changes the taste, your recipe isn't very stable. Similarly, if your model's predictions change
dramatically when you leave out just one data point, that's a sign your model might not be reliable.

This detector runs your model many times, each time leaving out a different data point, and analyzes
how consistent the results are. This helps identify:

- Overfitting: If your model performs much worse when certain points are left out
- Underfitting: If your model performs consistently poorly regardless of which points are left out
- Influential outliers: Individual data points that have an unusually large impact on your model

The jackknife approach is particularly useful for smaller datasets where you can't afford to set
aside a large validation set.

## How It Works

The jackknife technique (also known as "leave-one-out") involves systematically leaving out one
observation at a time from the dataset, retraining the model on the remaining data, and evaluating
its performance. By analyzing how model performance varies when different observations are excluded,
the detector can assess model stability and identify potential overfitting or underfitting issues.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxIterations` | Gets or sets the maximum number of jackknife iterations to perform. |
| `MinSampleSize` | Gets or sets the minimum sample size required to perform jackknife analysis. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting based on the jackknife analysis results. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting based on the jackknife analysis results. |

