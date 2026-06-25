---
title: "GaussianProcessFitDetectorOptions"
description: "Configuration options for the Gaussian Process Fit Detector, which analyzes model fit quality using Gaussian Process regression to detect overfitting, underfitting, and uncertainty issues."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Gaussian Process Fit Detector, which analyzes model fit quality
using Gaussian Process regression to detect overfitting, underfitting, and uncertainty issues.

## For Beginners

Think of this as a quality control tool that examines how well your
model's predictions match the actual data. It uses a special technique called Gaussian Process
regression that's particularly good at detecting patterns and measuring uncertainty. This detector
looks at the errors your model makes and determines whether they seem random (which is good) or
show patterns (which suggests problems). It can tell you if your model is too complex and memorizing
the data instead of learning general patterns (overfitting), or if it's too simple and missing
important patterns (underfitting). It can also identify areas where your model is particularly
uncertain about its predictions.

## How It Works

Gaussian Process Fit Detector uses Gaussian Process regression to analyze the residuals (differences
between predicted and actual values) of a model. By examining patterns in these residuals, it can
detect issues like overfitting (model captures noise rather than true patterns), underfitting
(model fails to capture important patterns), and areas of high uncertainty.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for considering model fit as good based on normalized residual patterns. |
| `HighUncertaintyThreshold` | Gets or sets the threshold for considering prediction uncertainty as high. |
| `LengthScale` | Gets or sets the length scale parameter for the Gaussian Process kernel. |
| `LowUncertaintyThreshold` | Gets or sets the threshold for considering prediction uncertainty as low. |
| `NoiseVariance` | Gets or sets the assumed noise variance in the data. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting based on residual patterns. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting based on residual patterns. |

