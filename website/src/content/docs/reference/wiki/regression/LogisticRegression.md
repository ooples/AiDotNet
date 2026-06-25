---
title: "LogisticRegression"
description: "Represents a logistic regression model for binary classification problems."
section: "Reference"
---

_Regression Models_

Represents a logistic regression model for binary classification problems.

## For Beginners

Logistic regression is like a decision-maker that predicts whether something belongs to one category or another. Think of it like determining whether an email is spam or not: - The model looks at different "features" of the email (like certain words or sender information) - It calculates how much each feature suggests "spam" or "not spam" - It combines all this information to make a final prediction between 0 and 1 - Values closer to 1 mean "more likely spam", values closer to 0 mean "more likely not spam" For example, words like "free" or "offer" might increase the spam probability, while emails from your contacts might decrease it. Logistic regression finds the right balance of these factors to make accurate predictions.

## How It Works

Logistic regression is a statistical method used for binary classification tasks, where the goal is to predict one of two possible outcomes (such as yes/no, true/false, 0/1). Unlike linear regression, which predicts continuous values, logistic regression outputs probabilities between 0 and 1, which can be interpreted as the likelihood of belonging to the positive class.

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
    .ConfigureModel(new LogisticRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained LogisticRegression.");
```

