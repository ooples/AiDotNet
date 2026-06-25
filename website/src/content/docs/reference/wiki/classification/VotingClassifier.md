---
title: "VotingClassifier<T>"
description: "Voting classifier that combines multiple classifiers through voting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Meta`

Voting classifier that combines multiple classifiers through voting.

## For Beginners

Voting combines predictions from multiple models:

Hard Voting:

- Each classifier votes for a class
- The class with most votes wins
- Example: [A, A, B] -> A wins (2 vs 1)

Soft Voting:

- Average the probability predictions
- Pick the class with highest average probability
- Generally works better when classifiers output calibrated probabilities

When to use:

- To combine different types of classifiers
- When you want to reduce the risk of a single bad model
- To leverage the strengths of different algorithms

## How It Works

Voting classifier combines predictions from multiple different classifiers
using either hard voting (majority vote) or soft voting (average probabilities).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VotingClassifier` | Initializes a new instance with default settings using Gaussian Naive Bayes estimators. |
| `VotingClassifier(IEnumerable<IClassifier<>>,VotingClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the VotingClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the voting-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `HardVotePredict(Matrix<>)` | Performs hard voting prediction. |
| `Predict(Matrix<>)` |  |
| `PredictLogProbabilities(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_estimators` | The list of classifiers in the ensemble. |
| `_weights` | The weights for each classifier. |

