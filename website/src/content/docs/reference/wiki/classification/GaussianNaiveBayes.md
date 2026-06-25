---
title: "GaussianNaiveBayes<T>"
description: "Gaussian Naive Bayes classifier for continuous features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.NaiveBayes`

Gaussian Naive Bayes classifier for continuous features.

## For Beginners

This classifier works well with continuous data (like measurements: height, weight, temperature).
It assumes each feature follows a bell-shaped curve (normal distribution) for each class.

During training, it learns:

- The average value of each feature for each class
- How spread out (variance) each feature is for each class

During prediction, it calculates how likely a new data point is under each class's
distribution and picks the most likely class.

Example use cases:

- Classifying iris flowers based on petal/sepal measurements
- Medical diagnosis based on patient vitals
- Weather prediction based on sensor readings

## How It Works

Gaussian Naive Bayes assumes that the continuous features follow a Gaussian (normal)
distribution within each class. It estimates the mean and variance of each feature
for each class during training, then uses these to compute the probability density
during prediction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianNaiveBayes(NaiveBayesOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the GaussianNaiveBayes class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep clone of this model. |
| `ComputeClassParameters(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `ComputeLogLikelihood(Vector<>,Int32)` | Computes the log-likelihood of a sample given a class using Gaussian distribution. |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Serialize` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_log2Pi` | Precomputed log(2 * pi) for efficiency in log-likelihood calculation. |
| `_means` | Mean values for each feature in each class. |
| `_variances` | Variance values for each feature in each class. |

