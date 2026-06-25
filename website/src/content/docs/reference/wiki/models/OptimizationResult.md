---
title: "OptimizationResult<T, TInput, TOutput>"
description: "Represents the comprehensive results of an optimization process for a symbolic model, including the best solution found, performance metrics, feature selection results, and detailed statistics for different datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the comprehensive results of an optimization process for a symbolic model, including the best solution found,
performance metrics, feature selection results, and detailed statistics for different datasets.

## For Beginners

This class stores everything about an optimization process and its results.

When optimizing a model:

- You start with an initial model and try to improve it
- You track how the model performs as it evolves
- You need to know which features were important
- You want to see how well it performs on different datasets

This class stores all that information, including:

- The best model found during optimization
- How good that model is (fitness score)
- How many iterations the optimization process ran
- How the model improved over time
- Which input features were selected
- Detailed performance metrics on training, validation, and test data
- Analysis of potential issues like overfitting

Having all this information in one place makes it easier to understand,
evaluate, and document your optimization results.

## How It Works

This class encapsulates all the information produced during the optimization of a symbolic model. It includes the best 
model found, its fitness score, the number of iterations performed, the history of fitness scores during optimization, 
the features selected for the model, detailed results for training, validation, and test datasets, fit detection 
analysis, and coefficient bounds. This comprehensive collection of information allows for thorough analysis of the 
optimization process and the resulting model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OptimizationResult` | Initializes a new instance of the OptimizationResult class with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestFitnessScore` | Gets or sets the fitness score of the best solution. |
| `BestIntercept` | Gets or sets the intercept term of the best solution. |
| `BestSolution` | Gets or sets the best model found during optimization. |
| `CoefficientLowerBounds` | Gets or sets the lower bounds for model coefficients. |
| `CoefficientUpperBounds` | Gets or sets the upper bounds for model coefficients. |
| `FitDetectionResult` | Gets or sets the results of fit detection analysis. |
| `FitnessHistory` | Gets or sets the history of fitness scores during optimization. |
| `Iterations` | Gets or sets the number of iterations performed during optimization. |
| `SelectedFeatureIndices` | Gets or sets the column indices of features selected during optimization. |
| `SelectedFeatures` | Gets or sets the list of feature vectors selected for the model. |
| `TestResult` | Gets or sets the detailed results for the test dataset. |
| `TrainingResult` | Gets or sets the detailed results for the training dataset. |
| `ValidationResult` | Gets or sets the detailed results for the validation dataset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` | Creates a deep copy of this OptimizationResult instance. |
| `WithParameters(Vector<>)` | Creates a new OptimizationResult instance with the best solution updated to use the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Provides numeric operations for the generic type T. |

