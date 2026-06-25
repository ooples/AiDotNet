---
title: "ProbabilisticClassifierBase<T>"
description: "Provides a base implementation for probabilistic classification algorithms that output class probability estimates."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification`

Provides a base implementation for probabilistic classification algorithms that output
class probability estimates.

## For Beginners

Probabilistic classifiers don't just say "this is category A" - they tell you how confident
they are. For example, instead of just "spam", they might say "92% spam, 8% not spam."

This additional information is valuable because:

- You can see when the model is uncertain (close to 50%/50%)
- You can adjust the decision threshold for your specific needs
- You can combine predictions from multiple models more effectively

## How It Works

This abstract class extends ClassifierBase to add probabilistic prediction capabilities.
Probabilistic classifiers can output not just the predicted class, but also the probability
of each class. This is useful for understanding model confidence and making threshold-based
decisions.

The default Predict() method uses argmax of the probabilities to determine the class.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProbabilisticClassifierBase(ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of the ProbabilisticClassifierBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySigmoid(Vector<>)` | Applies sigmoid function for binary classification probabilities. |
| `ApplySoftmax(Matrix<>)` | Applies softmax normalization to convert raw scores to probabilities. |
| `GetModelMetadata` |  |
| `Predict(Matrix<>)` | Predicts class labels for the given input data by taking the argmax of probabilities. |
| `PredictLogProbabilities(Matrix<>)` | Predicts log-probabilities for each class. |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for each sample in the input. |

