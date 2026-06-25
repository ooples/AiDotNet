---
title: "M5ModelTree"
description: "Represents an M5 model tree for regression problems, combining decision tree structure with linear models at the leaves."
section: "Reference"
---

_Regression Models_

Represents an M5 model tree for regression problems, combining decision tree structure with linear models at the leaves.

## For Beginners

An M5 model tree is like a smart decision-making system for predicting numbers. Think of it like a flowchart for home price prediction: - The tree asks questions about the home (Is it bigger than 2000 sq ft? Is it in neighborhood A?) - Based on the answers, you follow different paths down the tree - When you reach the end (a leaf), instead of getting a single price value, you get a mini-calculator (linear model) - This mini-calculator uses the home's features to make a more precise prediction for that specific group of homes For example, for small homes in urban areas, the price might depend more on location, while for large homes in suburbs, the number of bathrooms might be more important. The M5 model tree captures these different patterns for different groups of data.

## How It Works

The M5 model tree is an advanced regression technique that combines the benefits of decision trees and linear regression. Instead of using a single value at each leaf node (as in standard regression trees), M5 model trees fit linear regression models at each leaf. This allows the tree to capture both global patterns through its structure and local patterns through the linear models, often resulting in more accurate predictions compared to standard regression trees.

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
    .ConfigureModel(new M5ModelTree<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained M5ModelTree.");
```

