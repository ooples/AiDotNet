---
title: "GeneticAlgorithmRegression<T>"
description: "Implements a regression model that uses genetic algorithms to optimize model parameters, mimicking the process of natural selection to find the best solution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements a regression model that uses genetic algorithms to optimize model parameters,
mimicking the process of natural selection to find the best solution.

## For Beginners

This model uses a technique inspired by natural evolution to find the best solution.

Think of it like breeding the best solution:

- Start with a random "population" of potential solutions (different sets of coefficients)
- Test how well each solution performs on your data (fitness evaluation)
- Keep the best solutions and let them "reproduce" to create new solutions
- Occasionally introduce random changes (mutations) to explore new possibilities
- Repeat this process over multiple "generations" until you find an excellent solution

The benefit of this approach is that it can find good solutions to complex problems
without getting stuck in suboptimal answers. It's similar to how nature evolves
successful organisms over time, but applied to finding the best mathematical model.

## How It Works

Genetic Algorithm Regression uses evolutionary principles to find optimal model coefficients.
It maintains a population of potential solutions (models) that evolve over generations through
selection, crossover, and mutation operations. This approach is particularly useful for complex
problems where traditional optimization methods might struggle, as it can effectively explore
large solution spaces and avoid local optima.

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
    .ConfigureModel(new GeneticAlgorithmRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained GeneticAlgorithmRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GeneticAlgorithmRegression(RegressionOptions<>,GeneticAlgorithmOptimizerOptions<,Matrix<>,Vector<>>,IRegularization<,Matrix<>,Vector<>>,IOutlierRemoval<,Matrix<>,Vector<>>,PreprocessingPipeline<,Matrix<>,Matrix<>>)` | Initializes a new instance of the `GeneticAlgorithmRegression` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Trains the Genetic Algorithm Regression model using the provided input features and target values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the GeneticAlgorithmRegression with the same configuration as the current instance. |
| `Deserialize(Byte[])` | Loads a previously serialized Genetic Algorithm Regression model from a byte array. |
| `Predict(Matrix<>)` | Predicts target values for the provided input features using the trained Genetic Algorithm Regression model. |
| `PrependInterceptColumn(Matrix<>)` | Prepends a column of 1s to the matrix for the intercept term. |
| `Serialize` | Serializes the Genetic Algorithm Regression model to a byte array for storage or transmission. |
| `UpdateCoefficientsAndIntercept` | Gets the model type of the Genetic Algorithm Regression model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bestModel` | The best model found by the genetic algorithm. |
| `_gaOptions` | Configuration options for the genetic algorithm optimizer. |
| `_optimizer` | The genetic algorithm optimizer that finds optimal model parameters. |
| `_outlierRemoval` | Component that identifies and removes outliers from the training data. |
| `_preprocessingPipeline` | Component that handles all data preprocessing steps before training. |

