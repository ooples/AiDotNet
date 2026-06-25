---
title: "OptimizationIterationInfo<T>"
description: "Represents information about a single iteration in an optimization process, including fitness and overfitting detection results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents information about a single iteration in an optimization process, including fitness and overfitting detection results.

## For Beginners

This class stores information about one step in a model training or optimization process.

When training machine learning models or optimizing algorithms:

- The process typically runs through many iterations (steps)
- You want to track how well the model is performing at each step
- You need to detect when to stop training to avoid overfitting

This class stores information about a single iteration, including:

- Which iteration number it is
- How good the solution is at this step (fitness)
- Whether overfitting has been detected

This information helps you monitor the optimization process,
decide when to stop, and analyze how the solution improved over time.

## How It Works

This class encapsulates information about a specific iteration in an optimization or training process. It stores the 
iteration number, the fitness value achieved at that iteration, and results from overfitting detection. This information 
is useful for tracking the progress of optimization algorithms, analyzing convergence behavior, and detecting when 
training should be stopped to prevent overfitting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OptimizationIterationInfo` | Initializes a new instance of the OptimizationIterationInfo class with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FitDetectionResult` | Gets or sets the result of overfitting detection for this iteration. |
| `Fitness` | Gets or sets the fitness value at this iteration. |
| `Iteration` | Gets or sets the iteration number. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_fitness` | The backing field for the Fitness property. |

