---
title: "PartialDependencePlotFitDetectorOptions"
description: "Configuration options for the Partial Dependence Plot Fit Detector, which uses partial dependence plots to evaluate model fit quality and detect overfitting or underfitting in machine learning models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Partial Dependence Plot Fit Detector, which uses partial dependence plots
to evaluate model fit quality and detect overfitting or underfitting in machine learning models.

## For Beginners

The Partial Dependence Plot Fit Detector helps identify if your model is learning properly
by examining how it responds to changes in individual input features.

Imagine you built a model to predict house prices based on features like square footage, location, and age:

- A partial dependence plot shows how the predicted price changes when you vary just one feature (like square footage)

while keeping all other features at their average values

These plots help reveal three common problems:

- Overfitting: When your model learns patterns that are too specific to your training data

(like a jagged, noisy relationship between square footage and price)

- Underfitting: When your model is too simple and misses important patterns

(like a flat line that shows no relationship between square footage and price)

- Good fit: When your model captures meaningful patterns without excess complexity

(like a smooth curve showing prices increasing with square footage, but at a decreasing rate)

This class lets you configure how the detector analyzes these plots to automatically
identify potential fitting problems in your models.

## How It Works

Partial Dependence Plots (PDPs) visualize the relationship between a target variable and a set of input 
features of interest, marginalizing over the values of all other input features. The Partial Dependence 
Plot Fit Detector leverages these plots to assess model quality by comparing the complexity and smoothness 
of the learned relationships. This approach provides valuable insights into whether a model has captured 
meaningful patterns (good fit), is too simplistic (underfit), or has learned noise in the training data 
(overfit). By analyzing the characteristics of these plots across different features, the detector can 
provide early warnings of potential modeling issues that might not be apparent from aggregate metrics alone.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumPoints` | Gets or sets the number of points to sample for generating each partial dependence plot. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting based on the variability in partial dependence plots. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting based on the flatness in partial dependence plots. |

