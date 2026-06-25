---
title: "ConfusionMatrix<T>"
description: "Represents a confusion matrix for evaluating the performance of a classification model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinearAlgebra`

Represents a confusion matrix for evaluating the performance of a classification model.

## For Beginners

A confusion matrix helps you understand how well your AI model is performing
when classifying data into categories.

## How It Works

A confusion matrix is a table that summarizes the prediction results of a classification model.
It supports both binary classification (2x2 matrix) and multi-class classification (NxN matrix).

**Binary Classification:** For 2-class problems, it shows four important numbers:

- True PositivesWhen your model correctly predicted "Yes" (e.g., correctly identified a cat as a cat)
- True NegativesWhen your model correctly predicted "No" (e.g., correctly identified a non-cat as not a cat)
- False PositivesWhen your model incorrectly predicted "Yes" (e.g., identified a dog as a cat) - also called a "Type I error"
- False NegativesWhen your model incorrectly predicted "No" (e.g., identified a cat as not a cat) - also called a "Type II error"

**Multi-Class Classification:** For 3+ class problems, the matrix is NxN where N is the number of classes.
Rows represent predicted classes, columns represent actual classes. Cell [i,j] contains the count of samples
predicted as class i but actually belonging to class j.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConfusionMatrix(,,,)` | Initializes a new instance of the `ConfusionMatrix` class with the specified values. |
| `ConfusionMatrix(Int32)` | Initializes a new instance of the `ConfusionMatrix` class with the specified dimension. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Accuracy` | Gets the accuracy of the classification model. |
| `ClassCount` | Gets the number of classes represented in the confusion matrix. |
| `F1Score` | Gets the F1 score of the classification model. |
| `FalseNegatives` | Gets the number of false negative predictions (incorrectly predicted negative cases). |
| `FalsePositives` | Gets the number of false positive predictions (incorrectly predicted positive cases). |
| `Precision` | Gets the precision of the classification model. |
| `Recall` | Gets the recall (sensitivity) of the classification model. |
| `Specificity` | Gets the specificity of the classification model. |
| `TrueNegatives` | Gets the number of true negative predictions (correctly predicted negative cases). |
| `TruePositives` | Gets the number of true positive predictions (correctly predicted positive cases). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateInstance(Int32,Int32)` | Creates a new instance of a matrix with the specified dimensions. |
| `GetAccuracy` | Gets the overall accuracy across all classes. |
| `GetCohenKappa` | Gets the Cohen's Kappa coefficient, measuring inter-rater agreement. |
| `GetF1Score(Int32)` | Gets the F1 score for a specific class. |
| `GetFalseNegatives(Int32)` | Gets the false negatives for a specific class. |
| `GetFalsePositives(Int32)` | Gets the false positives for a specific class. |
| `GetHammingLoss` | Gets the Hamming Loss, measuring the fraction of incorrect predictions. |
| `GetJaccardScore` | Gets the Jaccard Score (Jaccard Index) for classification. |
| `GetJaccardScore(Int32)` | Gets the Jaccard Score for a specific class. |
| `GetMacroF1Score` | Gets the macro-averaged F1 score across all classes. |
| `GetMacroPrecision` | Gets the macro-averaged precision across all classes. |
| `GetMacroRecall` | Gets the macro-averaged recall across all classes. |
| `GetMatthewsCorrelationCoefficient` | Gets the Matthews Correlation Coefficient (MCC) for classification. |
| `GetMicroF1Score` | Gets the micro-averaged F1 score across all classes. |
| `GetMicroPrecision` | Gets the micro-averaged precision across all classes. |
| `GetMicroRecall` | Gets the micro-averaged recall across all classes. |
| `GetPrecision(Int32)` | Gets the precision for a specific class. |
| `GetRecall(Int32)` | Gets the recall for a specific class. |
| `GetTrueNegatives(Int32)` | Gets the true negatives for a specific class. |
| `GetTruePositives(Int32)` | Gets the true positives for a specific class. |
| `GetWeightedF1Score` | Gets the weighted-averaged F1 score across all classes. |
| `GetWeightedPrecision` | Gets the weighted-averaged precision across all classes. |
| `GetWeightedRecall` | Gets the weighted-averaged recall across all classes. |
| `Increment(Int32,Int32)` | Increments the count for a specific prediction-actual class pair. |

