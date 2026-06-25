---
title: "SymbolicRegression<T>"
description: "Implements symbolic regression, which discovers mathematical expressions that best describe the relationship between input features and target values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements symbolic regression, which discovers mathematical expressions that best describe the relationship
between input features and target values. Unlike traditional regression methods, symbolic regression
can discover both the form of the equation and its parameters.

## For Beginners

Symbolic regression is like having an AI mathematician that invents formulas.

Think of it like this:

- Instead of you telling the computer what equation to use (like y = mx + b)
- The computer tries thousands of different formulas (like y = x², y = sin(x), etc.)
- It tests each formula to see how well it predicts your data
- It combines good formulas to make even better ones
- Eventually, it finds a formula that best explains your data

For example, when modeling how a plant grows, instead of assuming it follows a linear or
exponential pattern, symbolic regression might discover it follows a pattern like
"growth = sunlight² × water / (1 + temperature)".

## How It Works

Symbolic regression works by:

- Creating a population of mathematical expressions (typically as expression trees)
- Evolving these expressions using genetic programming techniques
- Evaluating expressions based on how well they fit the data
- Selecting the best expressions to create new generations
- Eventually converging on an optimal or near-optimal mathematical model

This approach can discover complex, nonlinear relationships without requiring the user
to specify the form of the equation in advance.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new SymbolicRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained SymbolicRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SymbolicRegression(SymbolicRegressionOptions,IRegularization<,Matrix<>,Vector<>>,IFitnessCalculator<,Matrix<>,Vector<>>,IFitDetector<,Matrix<>,Vector<>>,IOutlierRemoval<,Matrix<>,Vector<>>,PreprocessingPipeline<,Matrix<>,Matrix<>>)` | Creates a new symbolic regression model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestFitness` | Gets the fitness score of the best model discovered during optimization. |
| `BestModel` | Gets the best symbolic model discovered during optimization. |
| `ParameterCount` | Optimizes the symbolic regression model using the provided input data and target values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateInstance` | Returns the type identifier for this regression model. |
| `GetOptions` |  |
| `Predict(Matrix<>)` | Predicts target values for a matrix of input features. |
| `PredictSingle(Vector<>)` | Predicts a target value for a single input feature vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bestFitness` | The fitness score of the best model found. |
| `_bestModel` | The best symbolic model found during the optimization process. |
| `_fitDetector` | The component that detects when a satisfactory model has been found. |
| `_fitnessCalculator` | The calculator used to evaluate the fitness or quality of symbolic models. |
| `_optimizer` | The optimizer used to evolve and improve symbolic models. |
| `_options` | Configuration options for the symbolic regression model. |
| `_outlierRemoval` | The component responsible for identifying and removing outliers from the data. |
| `_preprocessingPipeline` | The preprocessing pipeline that handles data transformation before training. |

