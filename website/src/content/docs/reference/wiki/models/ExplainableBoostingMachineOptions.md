---
title: "ExplainableBoostingMachineOptions"
description: "Configuration options for Explainable Boosting Machine (EBM) models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Explainable Boosting Machine (EBM) models.

## For Beginners

EBM is one of the few machine learning models that gives you
both high accuracy AND full explainability. Unlike "black box" models where you can't
see why they made a prediction, EBM shows you exactly how each feature affects the
outcome through interpretable graphs called "shape functions."

For example, if you're predicting house prices, EBM will show you:

- How square footage affects price (e.g., "each 100 sqft adds $10,000")
- How age affects price (e.g., "older houses are worth less")
- How these factors combine

This transparency is crucial for healthcare, finance, and other domains where
you need to explain and justify predictions.

## How It Works

EBM is a glass-box machine learning model that maintains high accuracy while being
fully interpretable. It's based on Generalized Additive Models (GAMs) with boosting,
allowing you to see exactly how each feature contributes to predictions.

## Properties

| Property | Summary |
|:-----|:--------|
| `CyclicTraining` | Gets or sets whether to use cyclic training order. |
| `DetectInteractions` | Gets or sets whether to automatically detect and include pairwise interactions. |
| `EarlyStoppingRounds` | Gets or sets the number of early stopping rounds. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxBins` | Gets or sets the maximum number of bins for continuous features. |
| `MaxInteractionBins` | Gets or sets the maximum number of interaction terms to consider. |
| `MinSamplesPerBin` | Gets or sets the minimum samples required in each bin. |
| `NumberOfInnerIterations` | Gets or sets the number of inner boosting iterations per feature. |
| `NumberOfOuterIterations` | Gets or sets the number of outer boosting iterations. |
| `RegularizationStrength` | Gets or sets the regularization strength for smoothing shape functions. |
| `SubsampleRatio` | Gets or sets the subsampling ratio for each iteration. |
| `ValidationFraction` | Gets or sets the fraction of data to use for validation during early stopping. |

