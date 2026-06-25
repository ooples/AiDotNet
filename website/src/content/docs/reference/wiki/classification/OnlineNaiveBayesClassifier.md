---
title: "OnlineNaiveBayesClassifier<T>"
description: "Implements Online (Incremental) Naive Bayes classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Online`

Implements Online (Incremental) Naive Bayes classifier.

## For Beginners

Online Naive Bayes is an incremental version of the classic
Naive Bayes classifier that can learn from streaming data. It maintains running statistics
(mean, variance, counts) that are updated with each new sample.

## How It Works

**How it works:**

- For each sample, update per-class feature statistics (mean, variance, count)
- To predict, calculate P(class|features) using Bayes' rule
- Assumes features are conditionally independent given the class
- Uses Gaussian distribution assumption for continuous features

**Key formulas:**

- P(class|x) ∝ P(class) × ∏ P(xi|class)
- P(xi|class) = N(xi; μ, σ²) for Gaussian assumption
- Online mean update: μ_new = μ_old + (x - μ_old) / n
- Online variance update: Welford's algorithm

**Advantages:**

- Very fast updates (constant time per sample)
- Low memory usage (only store statistics, not data)
- Works well with high-dimensional data
- Naturally handles class imbalance through prior updates

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlineNaiveBayesClassifier(OnlineNaiveBayesOptions<>)` | Creates a new Online Naive Bayes classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsWarm` | Gets whether the model has seen at least one sample. |
| `SamplesSeen` | Gets the total number of samples the model has seen. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` | Deserializes the trained model state including per-class statistics. |
| `GetOptions` |  |
| `GetParameters` |  |
| `PartialFit(Matrix<>,Vector<>)` | Updates the model with a batch of training samples. |
| `PartialFit(Vector<>,)` | Updates the model with a single training sample. |
| `Predict(Matrix<>)` |  |
| `PredictProbabilities(Vector<>)` | Gets the probability distribution over classes for a sample. |
| `Serialize` | Serializes the trained model state including per-class statistics. |
| `SetParameters(Vector<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

