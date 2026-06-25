---
title: "KernelRidgeRegression<T>"
description: "Implements Kernel Ridge Regression, a powerful nonlinear regression technique that combines ridge regression with the kernel trick to capture complex nonlinear relationships."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements Kernel Ridge Regression, a powerful nonlinear regression technique that combines
ridge regression with the kernel trick to capture complex nonlinear relationships.

## For Beginners

Kernel Ridge Regression is like using a special lens that helps see complex patterns in your data.

Regular linear models can only fit straight lines to data, but many real-world relationships aren't straight.
Kernel Ridge Regression solves this by:

- Using "kernels" to transform your data into a format where complex relationships become simpler
- Finding patterns in this transformed space
- Adding "ridge" regularization to prevent the model from becoming too complex or overfitting

Think of it like this: If you tried to separate red and blue dots on a sheet of paper with a single line,
sometimes it's impossible. But if you could lift some dots off the page (into 3D space), you might be able
to separate them with a flat plane. Kernels do something similar - they transform your data so that complex
patterns become easier to find.

This technique is particularly good for:

- Medium-sized datasets with complex relationships
- Problems where the relationship between inputs and outputs is highly nonlinear
- When you need both good prediction accuracy and the ability to adjust how much the model fits to noise

## How It Works

Kernel Ridge Regression extends linear ridge regression by applying the "kernel trick" to implicitly
map the input features to a higher-dimensional space without explicitly computing the transformation.
This allows the model to capture complex nonlinear relationships while still maintaining the computational
efficiency of ridge regression. The regularization parameter (lambda) helps prevent overfitting by
penalizing large coefficients.

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
    .ConfigureModel(new KernelRidgeRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained KernelRidgeRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KernelRidgeRegression` | Initializes a new instance with default settings. |
| `KernelRidgeRegression(KernelRidgeRegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `KernelRidgeRegression` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the configuration options specific to Kernel Ridge Regression. |
| `ParameterCount` | KRR doesn't benefit from optimizer parameter injection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a new instance of the KernelRidgeRegression with the same configuration as the current instance. |
| `Deserialize(Byte[])` | Loads a previously serialized Kernel Ridge Regression model from a byte array. |
| `ExtractModelParameters` | KRR uses all training points — skip SVR-style sparsification. |
| `GetModelMetadata` | Gets metadata about the Kernel Ridge Regression model and its configuration. |
| `OptimizeModel(Matrix<>,Vector<>)` | Optimizes the Kernel Ridge Regression model based on the provided training data. |
| `PredictSingle(Vector<>)` | Predicts the target value for a single input feature vector. |
| `Serialize` | Serializes the Kernel Ridge Regression model to a byte array for storage or transmission. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dualCoefficients` | The dual coefficients used for making predictions. |
| `_gramMatrix` | The Gram matrix (kernel matrix) that represents pairwise similarities between all training points. |

