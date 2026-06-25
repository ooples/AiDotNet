---
title: "MultinomialLogisticRegression<T>"
description: "Represents a multinomial logistic regression model for multi-class classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Represents a multinomial logistic regression model for multi-class classification problems.

## For Beginners

Multinomial logistic regression is a method for classifying data into multiple categories.

Think of it like a voting system where:

- Each feature (input variable) gets to "vote" for different categories
- The importance of each feature's vote is learned from training data
- For any new data point, we count the weighted votes for each category
- The category with the most votes wins and becomes the prediction

For example, when classifying emails into categories like "work," "personal," or "spam,"
certain words might strongly suggest one category over others. The model learns which
features (words) are most helpful for distinguishing between the different categories.

## How It Works

Multinomial logistic regression extends binary logistic regression to handle multiple classes. It models the probabilities
of different possible outcomes using the softmax function. For each class, the model learns a set of coefficients that
determine how each feature affects the probability of that class. During prediction, it assigns the input to the class
with the highest probability.

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
    .ConfigureModel(new MultinomialLogisticRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained MultinomialLogisticRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultinomialLogisticRegression(MultinomialLogisticRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `MultinomialLogisticRegression` class with optional custom options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Multinomial logistic is a classification model — no optimizer parameter injection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Matrix<>,Vector<>,Matrix<>)` | Computes the gradient of the log-likelihood with respect to the coefficients. |
| `ComputeHessian(Matrix<>,Matrix<>)` | Computes the Hessian matrix of the log-likelihood with respect to the coefficients. |
| `ComputeProbabilities(Matrix<>)` | Computes the probabilities of each class for each sample using the softmax function. |
| `CreateNewInstance` | Creates a new instance of the Multinomial Logistic Regression model with the same configuration. |
| `CreateOneHotEncoding(Vector<>)` | Creates a one-hot encoding of the class labels. |
| `Deserialize(Byte[])` | Deserializes the multinomial logistic regression model from a byte array. |
| `GetOptions` |  |
| `HasConverged(Matrix<>)` | Determines if the training has converged based on the magnitude of the coefficient updates. |
| `Predict(Matrix<>)` | Predicts the class labels for new data points using the trained multinomial logistic regression model. |
| `PredictProbabilities(Matrix<>)` | Predicts the probabilities of each class for new data points. |
| `Serialize` | Serializes the multinomial logistic regression model to a byte array for storage or transmission. |
| `Train(Matrix<>,Vector<>)` | Trains the multinomial logistic regression model using the provided features and target values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The coefficients matrix, where each row corresponds to a class and each column to a feature (plus intercept). |
| `_numClasses` | The number of distinct classes in the training data. |
| `_options` | The configuration options for the multinomial logistic regression model. |

