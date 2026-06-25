---
title: "MultinomialLogisticRegression"
description: "Represents a multinomial logistic regression model for multi-class classification problems."
section: "Reference"
---

_Regression Models_

Represents a multinomial logistic regression model for multi-class classification problems.

## For Beginners

Multinomial logistic regression is a method for classifying data into multiple categories. Think of it like a voting system where: - Each feature (input variable) gets to "vote" for different categories - The importance of each feature's vote is learned from training data - For any new data point, we count the weighted votes for each category - The category with the most votes wins and becomes the prediction For example, when classifying emails into categories like "work," "personal," or "spam," certain words might strongly suggest one category over others. The model learns which features (words) are most helpful for distinguishing between the different categories.

## How It Works

Multinomial logistic regression extends binary logistic regression to handle multiple classes. It models the probabilities of different possible outcomes using the softmax function. For each class, the model learns a set of coefficients that determine how each feature affects the probability of that class. During prediction, it assigns the input to the class with the highest probability.

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

