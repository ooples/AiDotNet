---
title: "FitnessCalculatorType"
description: "Specifies different loss functions and fitness calculators for evaluating model performance."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies different loss functions and fitness calculators for evaluating model performance.

## For Beginners

Loss functions and fitness calculators measure how well your AI model's predictions match the actual data.

Think of these metrics like grades on a test:

- They tell you how well your model is performing
- Different metrics focus on different aspects of performance
- For most loss functions, lower values mean better performance
- For some metrics (like R-squared), higher values mean better performance

When building AI models, you need ways to:

- Compare different models to choose the best one
- Know when to stop training your model
- Understand if your model is actually learning useful patterns
- Detect if your model is overfitting (memorizing data instead of learning)

Different metrics are better for different situations, so it's common to look at multiple metrics
when evaluating a model.

## Fields

| Field | Summary |
|:-----|:--------|
| `AdjustedRSquared` | A modified version of R-squared that adjusts for the number of predictors in the model. |
| `BinaryCrossEntropy` | Measures the cross-entropy loss for binary classification problems. |
| `CategoricalCrossEntropy` | Measures the cross-entropy loss for multi-class classification problems. |
| `Custom` | A custom loss function or fitness calculator defined by the user. |
| `ExponentialLoss` | Calculates the exponential loss, which heavily penalizes large errors. |
| `HuberLoss` | Calculates the Huber loss, which combines properties of MSE and MAE. |
| `LogCosh` | Calculates the logarithm of the hyperbolic cosine of the prediction error. |
| `MaxError` | Measures the maximum deviation between predicted and actual values. |
| `MeanAbsoluteError` | Calculates the average of the absolute differences between predicted and actual values. |
| `MeanAbsolutePercentageError` | Measures the mean absolute percentage error. |
| `MeanSquaredError` | Calculates the average of the squared differences between predicted and actual values. |
| `MeanSquaredLogError` | Calculates the mean squared logarithmic error. |
| `MedianAbsoluteError` | Measures the median absolute error between predicted and actual values. |
| `OrdinalRegressionLoss` | A loss function specifically designed for ordinal regression problems. |
| `RSquared` | Measures the proportion of variance in the dependent variable explained by the independent variables. |
| `RootMeanSquaredError` | Calculates the root mean squared error. |

