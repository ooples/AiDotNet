---
title: "GeneralizedAdditiveModel<T>"
description: "Implements a Generalized Additive Model (GAM) for regression, which models the target as a sum of smooth functions of individual features, allowing for flexible nonlinear relationships while maintaining interpretability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GeneralizedAdditiveModel(GeneralizedAdditiveModelOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `GeneralizedAdditiveModel` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Trains the Generalized Additive Model using the provided input features and target values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportances` | Gets the model type of the Generalized Additive Model. |
| `CreateBasisFunctions(Matrix<>,Boolean)` | Creates basis functions for each feature in the input data. |
| `CreateKnots(Vector<>)` | Creates knot points for spline basis functions based on the feature values. |
| `CreateNewInstance` | Creates a new instance of the GeneralizedAdditiveModel with the same configuration as the current instance. |
| `CreateSpline(Vector<>,,Int32)` | Creates a spline basis function for a feature using the specified knot and degree. |
| `Deserialize(Byte[])` | Loads a previously serialized Generalized Additive Model from a byte array. |
| `FitModel(Vector<>)` | Fits the model coefficients using the basis functions and target values. |
| `GetModelMetadata` | Gets metadata about the Generalized Additive Model and its configuration. |
| `GetOptions` |  |
| `Predict(Matrix<>)` | Predicts target values for the provided input features using the trained Generalized Additive Model. |
| `Serialize` | Serializes the Generalized Additive Model to a byte array for storage or transmission. |
| `SplineFunction(,,Int32)` | Computes the spline function value for a given input, knot, and degree. |
| `ValidateInputs(Matrix<>,Vector<>)` | Validates that the input data dimensions are compatible. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_basisFunctions` | Matrix of basis functions applied to the input features. |
| `_coefficients` | Vector of model coefficients for the basis functions. |
| `_options` | Configuration options for the Generalized Additive Model. |
| `_trainingKnots` | Knots fitted from the TRAINING feature distribution, one knot vector per feature column. |
| `_useOLS` | Tracks whether OLS fallback was used. |

