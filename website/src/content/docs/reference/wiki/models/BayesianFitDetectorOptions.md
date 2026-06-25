---
title: "BayesianFitDetectorOptions"
description: "Configuration options for the Bayesian model fit detector, which evaluates how well a model fits the data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Bayesian model fit detector, which evaluates how well a model fits the data.

## For Beginners

When building AI models, it's important to know if your model is "just right" 
for your data. Think of it like Goldilocks choosing a bed - one can be too soft (overfit), one too hard (underfit), 
and one just right (good fit). This class helps set the thresholds for determining which category your model falls into.

## How It Works

This class provides threshold values used to interpret Bayesian Information Criterion (BIC) or similar
Bayesian metrics that assess model fit. These thresholds help determine if a model is a good fit,
overfit (too complex), or underfit (too simple) for the given data.

An overfit model is like memorizing exam answers without understanding the concepts - it works perfectly for the 
practice questions but fails on the actual exam. An underfit model is too simple, like using a straight line to 
predict stock prices that go up and down. A good fit balances complexity and generalization, capturing the important 
patterns without getting distracted by random noise in the data.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for determining a good model fit. |
| `OverfitThreshold` | Gets or sets the threshold for detecting model overfitting. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting model underfitting. |

