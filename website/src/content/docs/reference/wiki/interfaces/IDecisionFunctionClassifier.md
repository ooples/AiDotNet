---
title: "IDecisionFunctionClassifier<T>"
description: "Interface for classifiers that compute a decision function for predictions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for classifiers that compute a decision function for predictions.

## For Beginners

Think of classification as drawing a line (or surface) to separate classes.
The decision function tells you how far a point is from that line:

- Positive values: On the positive class side
- Negative values: On the negative class side
- Values near zero: Close to the decision boundary (uncertain)

For example, in spam detection:

- Decision value +3.5: Strongly predicted as spam
- Decision value +0.2: Weakly predicted as spam
- Decision value -0.1: Weakly predicted as not spam
- Decision value -2.8: Strongly predicted as not spam

This is different from probabilities (which range from 0 to 1).
Decision values can be any real number.

## How It Works

Some classifiers, particularly Support Vector Machines, make predictions based on a
decision function that measures the "confidence" or "distance" from the decision boundary.
This interface provides access to these raw decision values.

## Properties

| Property | Summary |
|:-----|:--------|
| `NSupportVectors` | Gets the number of support vectors. |
| `SupportVectors` | Gets the support vectors learned during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DecisionFunction(Matrix<>)` | Computes the decision function for the input samples. |

