---
title: "ExtremelyRandomizedTreesRegression"
description: "Implements an Extremely Randomized Trees regression model, which is an ensemble method that uses multiple decision trees with additional randomization for improved prediction accuracy and reduced overfitting."
section: "Reference"
---

_Regression Models_

Implements an Extremely Randomized Trees regression model, which is an ensemble method that uses multiple decision trees
with additional randomization for improved prediction accuracy and reduced overfitting.

## For Beginners

This model works like a committee of decision trees that vote on predictions.

While a single decision tree might make mistakes due to its specific structure, 
a group of different trees can work together to make more reliable predictions:

- Each tree sees a random subset of the training data
- Each tree uses random thresholds for making decisions
- The final prediction is the average of all individual tree predictions

The key advantage is that by adding extra randomness in how the trees are built,
the model avoids "memorizing" the training data and becomes better at generalizing
to new data. This is similar to how asking many different people for their opinion
often leads to better decisions than relying on just one person.

## How It Works

Extremely Randomized Trees (also known as Extra Trees) is an ensemble method that builds multiple decision trees
and averages their predictions to improve accuracy and reduce overfitting. Unlike Random Forests, which use the best
split for each feature, Extra Trees selects random thresholds for each feature and chooses the best among these
random thresholds, adding an additional layer of randomization that can further reduce variance.

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
    .ConfigureModel(new ExtremelyRandomizedTreesRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained ExtremelyRandomizedTreesRegression.");
```

