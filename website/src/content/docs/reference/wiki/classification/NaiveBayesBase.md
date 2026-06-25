---
title: "NaiveBayesBase<T>"
description: "Provides a base implementation for Naive Bayes classifiers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification.NaiveBayes`

Provides a base implementation for Naive Bayes classifiers.

## For Beginners

Naive Bayes uses probability to make predictions. It learns from training data:

1. How common each class is (prior probability)
2. How likely each feature value is given each class (likelihood)

Then for a new sample, it calculates: P(class|features) ∝ P(class) × P(features|class)
and picks the class with the highest probability.

## How It Works

Naive Bayes classifiers are probabilistic classifiers based on Bayes' theorem with
strong (naive) independence assumptions between the features. Despite these assumptions,
Naive Bayes classifiers often perform very well in practice.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NaiveBayesBase(NaiveBayesOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the NaiveBayesBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassCounts` | Stores the count of samples per class during training. |
| `LogPriors` | Stores the log prior probabilities for each class. |
| `Options` | Gets the Naive Bayes specific options. |
| `SupportsParameterInitialization` | Naive Bayes models compute parameters from class statistics during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ComputeClassParameters(Matrix<>,Vector<>)` | Computes class-specific parameters during training. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `ComputeLogLikelihood(Vector<>,Int32)` | Computes the log-likelihood of a sample given a class. |
| `ComputeLogPriors(Int32)` | Computes the log prior probabilities for each class. |
| `GetClassIndex()` | Gets the class index for a given label value. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `LogSumExp(Vector<>)` | Computes the log-sum-exp for numerical stability. |
| `PredictLogProbabilities(Matrix<>)` | Predicts log-probabilities for each class (more numerically stable than probabilities). |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for each sample. |
| `SetParameters(Vector<>)` |  |
| `Train(Matrix<>,Vector<>)` | Trains the Naive Bayes classifier on the provided data. |
| `WithParameters(Vector<>)` |  |

