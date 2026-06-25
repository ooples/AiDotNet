---
title: "CrossValidationFitDetectorOptions"
description: "Configuration options for detecting overfitting, underfitting, and good fitting in machine learning models using cross-validation techniques."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for detecting overfitting, underfitting, and good fitting in machine learning models
using cross-validation techniques.

## For Beginners

Think of these settings as the criteria for judging how well your AI model has learned.
Just like when learning a new skill, an AI can learn too little (underfit), memorize without understanding (overfit),
or learn just right (good fit). These thresholds help determine which category your model falls into by comparing
how it performs on data it has seen before (training data) versus new data (validation data).

## How It Works

The CrossValidationFitDetectorOptions class provides threshold settings that help determine whether a model
is overfitting (performing well on training data but poorly on validation data), underfitting (performing
poorly on both training and validation data), or has a good fit (performing well on both).

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for identifying a good fit in a model. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting in a model. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting in a model. |

