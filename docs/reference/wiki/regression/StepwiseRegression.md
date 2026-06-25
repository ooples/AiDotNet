---
title: "StepwiseRegression"
description: "Implements stepwise regression, which automatically selects the most relevant features for the model."
section: "Reference"
---

_Regression Models_

Implements stepwise regression, which automatically selects the most relevant features for the model.
This approach builds a model by adding or removing features based on their statistical significance.

## For Beginners

Stepwise regression is like a smart shopping assistant that helps you 
pick only the most useful ingredients for a recipe.

Think of it like this:

- You have many potential ingredients (features) that might affect the outcome
- Instead of using all ingredients, which could make the recipe complicated or less tasty
- Stepwise regression tests each ingredient to see how much it improves the recipe
- It keeps only the ingredients that make a significant difference to the final result

For example, when predicting house prices, you might have data on square footage, 
number of bedrooms, location, age, etc. Stepwise regression would determine which of these 
features are most important for accurate predictions and discard the rest.

## How It Works

Stepwise regression helps solve the feature selection problem by iteratively building a model, either by:

- Forward selection: Starting with no features and adding the most significant ones
- Backward elimination: Starting with all features and removing the least significant ones

At each step, the algorithm evaluates the impact of adding or removing features based on a fitness metric
such as adjusted R-squared, AIC, BIC, or other statistical criteria.

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
    .ConfigureModel(new StepwiseRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained StepwiseRegression.");
```

