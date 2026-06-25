---
title: "AdaBoostR2Regression<T>"
description: "Implements the AdaBoost.R2 algorithm for regression problems, an ensemble learning method that combines multiple decision tree regressors to improve prediction accuracy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Implements the AdaBoost.R2 algorithm for regression problems, an ensemble learning method that combines
multiple decision tree regressors to improve prediction accuracy.

## For Beginners

AdaBoost.R2 is a powerful machine learning technique for predicting numeric values
(like prices, temperatures, or ages) rather than categories.

Think of AdaBoost.R2 as a team of experts (decision trees) working together to make predictions:

1. The first "expert" makes predictions on all the training data
2. The algorithm identifies which samples were predicted poorly
3. The next expert pays special attention to those difficult samples
4. This process repeats, creating a team of experts that each specialize in different aspects of the problem
5. When making predictions, all experts "vote" on the final answer, but experts who performed better get more voting power

This approach is particularly effective because:

- It can turn a collection of "weak" learners (simple decision trees) into a "strong" learner
- It automatically focuses on the hardest parts of the problem
- It's less prone to overfitting than a single, complex model

AdaBoost.R2 is ideal for problems where you need high prediction accuracy and have enough training data
to build multiple models.

## How It Works

AdaBoost.R2 (Adaptive Boosting for Regression) is an extension of the AdaBoost algorithm for regression tasks.
It works by training a sequence of weak regressors (decision trees) on repeatedly modified versions of the data.
The predictions from all regressors are then combined through a weighted majority vote to produce the final prediction.

In AdaBoost.R2, each training sample is assigned a weight that determines its importance during training.
Initially, all weights are equal. For each iteration, the weights of incorrectly predicted samples are increased
so that subsequent weak regressors focus more on difficult cases. The algorithm stops when the specified number
of estimators is reached or when the error rate exceeds 0.5.

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
    .ConfigureModel(new AdaBoostR2Regression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained AdaBoostR2Regression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaBoostR2Regression` | Initializes a new instance with default settings. |
| `AdaBoostR2Regression(AdaBoostR2RegressionOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `AdaBoostR2Regression` class with specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDepth` | Gets the maximum depth of each decision tree in the ensemble. |
| `NumberOfTrees` | Gets the number of decision trees in the ensemble. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateAverageError(Vector<>,Vector<>,,Vector<>)` | Calculates the absolute error between the target values and predictions. |
| `CalculateFeatureImportancesAsync(Int32)` | Updates the sample weights for the next iteration of AdaBoost.R2. |
| `CreateNewInstance` | Creates a new instance of the AdaBoostR2Regression with the same configuration as the current instance. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `GetModelMetadata` | Gets metadata about the trained model. |
| `PredictAsync(Matrix<>)` | Makes predictions on new data using the trained ensemble of decision trees asynchronously. |
| `Serialize` | Serializes the model to a byte array for storage or transmission. |
| `TrainAsync(Matrix<>,Vector<>)` | Trains the AdaBoost.R2 regression model on the provided input data and target values asynchronously. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_ensemble` | The ensemble of decision trees and their corresponding weights. |
| `_options` | Options for configuring the AdaBoost.R2 regression algorithm. |
| `_random` | Random number generator for creating diverse decision trees. |

