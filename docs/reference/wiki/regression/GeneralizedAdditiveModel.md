---
title: "GeneralizedAdditiveModel"
description: "Implements a Generalized Additive Model (GAM) for regression, which models the target as a sum of smooth functions of individual features, allowing for flexible nonlinear relationships while maintaining interpretability."
section: "Reference"
---

_Regression Models_

Implements a Generalized Additive Model (GAM) for regression, which models the target as a sum of smooth functions
of individual features, allowing for flexible nonlinear relationships while maintaining interpretability.

## For Beginners

A Generalized Additive Model is like a more flexible version of linear regression.

Instead of assuming that each feature has a straight-line relationship with the target (like y = mx + b),
GAMs allow each feature to have its own curved relationship with the target. The model then adds up
all these individual curves to make a prediction.

Think of it this way:

- Linear regression: House price = a × Size + b × Age + c × Location + ...
- GAM: House price = f1(Size) + f2(Age) + f3(Location) + ...

Where f1, f2, f3 are curves rather than straight lines

The benefit is that you can:

- Capture more complex patterns in your data
- Still understand how each feature individually affects the prediction
- Visualize the shape of the relationship for each feature

GAMs are a good middle ground between simple linear models and complex "black box" models
like neural networks.

## How It Works

Generalized Additive Models extend linear regression by allowing nonlinear relationships between features and the target
variable, while maintaining additivity. Each feature is transformed using basis functions (typically splines), and
the model is expressed as a sum of these transformations. This approach balances flexibility and interpretability,
as the effect of each feature can be visualized independently.

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
    .ConfigureModel(new GeneralizedAdditiveModel<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained GeneralizedAdditiveModel.");
```

