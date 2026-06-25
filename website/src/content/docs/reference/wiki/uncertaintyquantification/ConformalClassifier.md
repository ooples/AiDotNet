---
title: "ConformalClassifier<T>"
description: "Implements Conformal Prediction for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.UncertaintyQuantification.ConformalPrediction`

Implements Conformal Prediction for classification tasks.

## For Beginners

Conformal Prediction for classification provides prediction sets
with guaranteed coverage, not just a single class prediction.

Key differences from regression conformal prediction:

- Instead of an interval, you get a SET of possible classes
- The set is guaranteed to contain the true class with specified probability

Example with 90% confidence:

- Traditional classifier: "This is a cat" (might be wrong)
- Conformal classifier: "This is {cat, dog}" (90% guaranteed to include correct class)

Benefits:

- When the model is uncertain, you get a larger prediction set (e.g., {cat, dog, rabbit})
- When the model is confident, you get a smaller set (e.g., {cat})
- You can defer to a human expert when the prediction set is too large

This is invaluable for:

- Medical diagnosis: Know when to seek specialist opinion
- Autonomous systems: Know when to hand control back to operator
- Quality control: Flag uncertain cases for manual review

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConformalClassifier(INeuralNetwork<>,Int32)` | Initializes a new instance of the ConformalClassifier class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calibrate(Matrix<>,Vector<Int32>)` | Calibrates the conformal classifier using a calibration dataset. |
| `ComputeAverageSetSize(Matrix<>,Double)` | Computes the average size of prediction sets on a test set. |
| `ComputeThreshold(Double)` | Computes the threshold for prediction set inclusion. |
| `EvaluateCoverage(Matrix<>,Vector<Int32>,Double)` | Evaluates the empirical coverage on a test set. |
| `PredictSet(Tensor<>,Double)` | Predicts with a conformal prediction set. |

