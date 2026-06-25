---
title: "NegativeBinomialRegression<T>"
description: "Represents a negative binomial regression model for count data that may exhibit overdispersion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Represents a negative binomial regression model for count data that may exhibit overdispersion.

## For Beginners

Negative binomial regression is a special model for predicting counts when your data shows more variation than expected.

Think of it like predicting the number of customer service calls a business receives each day:

- A simple model might assume consistent variation around the average (Poisson model)
- But real data often shows much more variation - some days have way more calls than expected
- Negative binomial regression handles this "extra randomness" by including a special adjustment (dispersion parameter)

For example, the model might predict that a business receives 15 calls per day on average, but also accounts for the fact
that some days might have 5 calls while others have 40, which is more extreme variation than simpler models would expect.

## How It Works

Negative binomial regression is a type of generalized linear model used for modeling count data when the variance
exceeds the mean (overdispersion), which violates the assumption of Poisson regression. It extends Poisson regression
by adding a dispersion parameter that accounts for the extra variance in the data. The model uses a log link function
to ensure that predictions are always positive, as required for count data.

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
    .ConfigureModel(new NegativeBinomialRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained NegativeBinomialRegression.");
```

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateWeights(Vector<>)` | Calculates the weights for the iteratively reweighted least squares algorithm. |
| `CalculateWorkingResponse(Vector<>,Vector<>,Vector<>)` | Calculates the working response for the iteratively reweighted least squares algorithm. |
| `CreateNewInstance` | Creates a new instance of the Negative Binomial Regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the negative binomial regression model from a byte array. |
| `InitializeCoefficients(Int32)` | Initializes the model coefficients to zeros. |
| `Predict(Matrix<>)` | Makes predictions for new data points using the trained negative binomial regression model. |
| `Serialize` | Serializes the negative binomial regression model to a byte array for storage or transmission. |
| `Train(Matrix<>,Vector<>)` | Trains the negative binomial regression model using the provided features and target values. |
| `UpdateDispersion(Matrix<>,Vector<>)` | Updates the dispersion parameter based on the Pearson residuals. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dispersion` | The dispersion parameter that accounts for overdispersion in the data. |
| `_options` | The configuration options for the negative binomial regression model. |
| `_yShift` | Initializes a new instance of the `NegativeBinomialRegression` class with optional custom options and regularization. |

