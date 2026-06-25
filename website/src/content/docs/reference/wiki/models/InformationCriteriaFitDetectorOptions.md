---
title: "InformationCriteriaFitDetectorOptions"
description: "Configuration options for the Information Criteria Fit Detector, which uses statistical information criteria like AIC and BIC to evaluate model quality and complexity trade-offs."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Information Criteria Fit Detector, which uses statistical information
criteria like AIC and BIC to evaluate model quality and complexity trade-offs.

## For Beginners

This detector helps you choose the best model by balancing two competing
goals: how well the model fits your data and how simple the model is.

Think of it like shopping for a car. You want good performance (model fit), but you also care about
fuel efficiency (model simplicity). Information criteria like AIC and BIC give you a single score that
considers both aspects, helping you make better decisions.

Why is this important? Because a very complex model might fit your training data perfectly but perform
poorly on new data (like a gas-guzzling sports car that's impractical for daily use). On the other hand,
a model that's too simple might miss important patterns (like an underpowered car that can't handle hills).

The Information Criteria Fit Detector uses these scores to help you find the "sweet spot" - a model that's
just complex enough to capture the important patterns in your data, but no more complex than necessary.

## How It Works

Information criteria are statistical measures that balance model fit against complexity to help select
the most appropriate model. The two most common criteria are AIC (Akaike Information Criterion) and
BIC (Bayesian Information Criterion), which penalize models based on the number of parameters they use.
This helps prevent overfitting by favoring simpler models unless more complex ones provide significantly
better fit.

## Properties

| Property | Summary |
|:-----|:--------|
| `AicThreshold` | Gets or sets the threshold for significant differences in AIC (Akaike Information Criterion) values when comparing models. |
| `BicThreshold` | Gets or sets the threshold for significant differences in BIC (Bayesian Information Criterion) values when comparing models. |
| `HighVarianceThreshold` | Gets or sets the threshold for detecting high variance based on the relative difference between information criteria across different data samples. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting based on the relative difference between information criteria of nested models. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting based on the relative difference between information criteria of nested models. |

