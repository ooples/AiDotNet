---
title: "HistGradientBoostingRegression"
description: "Histogram-based Gradient Boosting Regression for fast training on large datasets."
section: "Reference"
---

_Regression Models_

Histogram-based Gradient Boosting Regression for fast training on large datasets.

## For Beginners

Traditional gradient boosting looks at every possible split point for every feature, which is slow for large datasets. Histogram-based methods group similar values into "bins" first, then only consider splits between bins. Think of it like sorting students by height: - Traditional method: Consider every student's exact height as a potential grouping point - Histogram method: First group students into height ranges (5'0"-5'2", 5'2"-5'4", etc.), then only consider splitting between groups This is much faster because there are far fewer groups than individual heights. Key advantages: - 10-100x faster than traditional gradient boosting on large datasets - Memory efficient (stores bin indices, not raw values) - Handles missing values naturally - Similar accuracy to traditional methods This is the same approach used by LightGBM, XGBoost (hist mode), and scikit-learn's HistGradientBoostingRegressor. Usage:

## How It Works

Histogram-based Gradient Boosting discretizes continuous features into a fixed number of bins, then builds histograms of gradients and hessians for each bin. This approach dramatically reduces the time complexity of finding the best split from O(n*features) to O(bins*features), making it suitable for large datasets with millions of samples.

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
    .ConfigureModel(new HistGradientBoostingRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained HistGradientBoostingRegression.");
```

