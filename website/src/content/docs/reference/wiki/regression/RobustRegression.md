---
title: "RobustRegression"
description: "Represents a robust regression model that is resistant to outliers in the data."
section: "Reference"
---

_Regression Models_

Represents a robust regression model that is resistant to outliers in the data.

## For Beginners

Traditional regression models can be heavily influenced by outliers 
(unusual data points that don't follow the general pattern). 

Think of robust regression like a smart voting system:

- It identifies which data points are "suspicious" (potential outliers)
- It gives these points less influence (lower weight) in determining the final model
- It focuses more on the reliable data points to find the true pattern

For example, if most houses in a neighborhood cost $200,000-300,000, but one special mansion costs 
$2 million, robust regression would recognize this as an outlier and reduce its influence when 
predicting house prices based on size or features.

## How It Works

Robust regression provides an alternative to traditional regression methods when data contains 
outliers or influential observations. By using weight functions, it reduces the influence of outliers 
on the final model. This implementation uses an iterative reweighted least squares approach to 
estimate coefficients that are less affected by extreme values.

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
    .ConfigureModel(new RobustRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained RobustRegression.");
```

