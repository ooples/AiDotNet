---
title: "OptimizationAlgorithmOptions<T, TInput, TOutput>"
description: "Configuration options for optimization algorithms used in machine learning models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for optimization algorithms used in machine learning models.

## For Beginners

Think of optimization as the process of "learning" in machine learning.
It's like adjusting the knobs on a radio until you get the clearest signal. These settings control
how quickly and accurately the algorithm learns from your data.

## How It Works

Optimization algorithms are methods used to find the best parameters for a machine learning model
by minimizing or maximizing an objective function.

## Properties

| Property | Summary |
|:-----|:--------|
| `BadFitPatience` | Gets or sets the number of iterations to wait before adjusting parameters when the model is performing poorly. |
| `EarlyStoppingPatience` | Gets or sets the number of iterations to wait before stopping if no improvement is observed. |
| `ExplorationRate` | Gets or sets the exploration rate for reinforcement learning and some optimization algorithms. |
| `FeatureSelectionProbability` | Gets or sets the probability of selecting a feature during feature selection mode. |
| `FitDetector` | Gets or sets the fit detector to determine when a model has converged or is overfitting. |
| `FitnessCalculator` | Gets or sets the fitness calculator to evaluate model quality during optimization. |
| `InitialLearningRate` | Gets or sets the initial learning rate for the optimization algorithm. |
| `InitialMomentum` | Gets or sets the initial momentum value. |
| `LearningRateDecay` | Gets or sets the rate at which the learning rate decreases over time. |
| `MaxExplorationRate` | Gets or sets the maximum allowed exploration rate. |
| `MaxIterations` | Gets or sets the maximum number of iterations (epochs) for the optimization algorithm. |
| `MaxLearningRate` | Gets or sets the maximum allowed learning rate. |
| `MaxMomentum` | Gets or sets the maximum allowed momentum value. |
| `MaximumFeatures` | Gets or sets the maximum number of features to consider in the model. |
| `MinExplorationRate` | Gets or sets the minimum allowed exploration rate. |
| `MinLearningRate` | Gets or sets the minimum allowed learning rate. |
| `MinMomentum` | Gets or sets the minimum allowed momentum value. |
| `MinimumFeatures` | Gets or sets the minimum number of features to consider in the model. |
| `ModelCache` | Gets or sets the model cache to store and retrieve previously evaluated models. |
| `ModelStatsOptions` | Gets or sets the options for model statistics calculation. |
| `MomentumDecreaseFactor` | Gets or sets the factor by which momentum decreases when performance worsens. |
| `MomentumIncreaseFactor` | Gets or sets the factor by which momentum increases when performance improves. |
| `OptimizationMode` | Gets or sets the optimization mode (feature selection, parameter tuning, or both). |
| `ParameterAdjustmentProbability` | Gets or sets the probability of adjusting a parameter during parameter tuning mode. |
| `ParameterAdjustmentScale` | Gets or sets the scale factor for parameter adjustments during optimization. |
| `PredictionOptions` | Gets or sets the options for prediction statistics calculation. |
| `SignFlipProbability` | Gets or sets the probability of flipping the sign of a parameter during perturbation. |
| `Tolerance` | Gets or sets the convergence tolerance for the optimization algorithm. |
| `UseAdaptiveLearningRate` | Gets or sets whether to automatically adjust the learning rate during training based on per-epoch fitness improvement (see `OptimizationStepData{`). |
| `UseAdaptiveMomentum` | Gets or sets whether to automatically adjust the momentum during training. |
| `UseEarlyStopping` | Gets or sets whether to use early stopping to prevent overfitting. |
| `UseExpressionTrees` | Gets or sets whether to use expression trees for optimization. |

