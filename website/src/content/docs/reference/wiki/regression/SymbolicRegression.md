---
title: "SymbolicRegression"
description: "Implements symbolic regression, which discovers mathematical expressions that best describe the relationship between input features and target values."
section: "Reference"
---

_Regression Models_

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

