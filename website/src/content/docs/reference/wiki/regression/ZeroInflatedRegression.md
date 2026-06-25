---
title: "ZeroInflatedRegression"
description: "Zero-Inflated regression for count data with excess zeros."
section: "Reference"
---

_Regression Models_

Zero-Inflated regression for count data with excess zeros.

## For Beginners

Imagine counting how many times customers visit a store each month:

- Some people NEVER visit (structural zeros) - they live far away or shop elsewhere
- Some people visit sometimes but happened to visit 0 times this month (sampling zeros)

Standard Poisson regression treats all zeros the same, but Zero-Inflated models
recognize these two types of zeros:

1. The "zero model" predicts WHO are structural zeros (π)
2. The "count model" predicts HOW MANY for non-structural-zero people (λ)

Example interpretation:

- "30% of potential customers are 'never visitors' (π = 0.3)"
- "Among potential visitors, the average visit rate is 2.5 times/month (λ = 2.5)"

This gives better predictions and allows you to understand both processes.

## How It Works

Zero-Inflated models handle count data where there are more zeros than a standard
count distribution would predict. They model the data as a mixture: with probability π,
the observation is a "structural zero," and with probability (1-π), it follows a count
distribution (Poisson or Negative Binomial).

Reference: Lambert, D. (1992). "Zero-Inflated Poisson Regression, with an Application
to Defects in Manufacturing". Technometrics, 34(1), 1-14.

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
    .ConfigureModel(new ZeroInflatedRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained ZeroInflatedRegression.");
```

