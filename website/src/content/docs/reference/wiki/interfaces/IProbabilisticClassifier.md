---
title: "IProbabilisticClassifier<T>"
description: "Defines the interface for classifiers that can output probability estimates for each class."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the interface for classifiers that can output probability estimates for each class.

## For Beginners

Some classifiers don't just say "this is category A" -
they also tell you how confident they are.

For example, when classifying an email as spam:

- A basic classifier might just say: "Spam"
- A probabilistic classifier says: "90% spam, 10% not spam"

The probability information is valuable because:

- You can see when the model is uncertain (50%/50% vs 99%/1%)
- You can adjust the decision threshold (e.g., only mark as spam if >95% confident)
- You can combine predictions from multiple models more effectively

Common probabilistic classifiers include:

- Naive Bayes (naturally outputs probabilities)
- Logistic Regression (outputs probabilities via sigmoid/softmax)
- Random Forest (outputs probabilities via vote counting)

## How It Works

Probabilistic classifiers extend the basic classification interface by providing
methods to obtain class probability estimates. This is useful for understanding
the model's confidence in its predictions and for decision-making that considers
uncertainty.

## Methods

| Method | Summary |
|:-----|:--------|
| `PredictLogProbabilities(Matrix<>)` | Predicts log-probabilities for each class. |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for each sample in the input. |

