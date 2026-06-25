---
title: "HybridFitDetector<T, TInput, TOutput>"
description: "A detector that combines multiple fit detection approaches to provide more robust model evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that combines multiple fit detection approaches to provide more robust model evaluation.

## For Beginners

This class combines two different ways of checking how well your model is performing:

1. Residual analysis - which looks at the differences between predicted and actual values
2. Learning curve analysis - which examines how your model performs as it sees more training data

By using both approaches together, this detector can give you more reliable insights about whether
your model is a good fit, overfitting, underfitting, or has other issues. Think of it like getting
a second opinion from another doctor - having multiple perspectives leads to better diagnosis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridFitDetector(ResidualAnalysisFitDetector<,,>,LearningCurveFitDetector<,,>,HybridFitDetectorOptions)` | Initializes a new instance of the HybridFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level in the fit type determination by combining confidence levels from multiple detection methods. |
| `CombineConfidenceLevels(,)` | Combines confidence levels from different detection methods into a single, weighted confidence score. |
| `CombineFitTypes(FitType,FitType)` | Combines fit types from different detection methods into a single, consensus fit type. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model by combining results from residual analysis and learning curve analysis. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type of a model by combining results from multiple detection methods. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_learningCurveDetector` | The learning curve component of the hybrid detector. |
| `_options` | Configuration options that control how the hybrid detector combines and weighs results from its components. |
| `_residualAnalyzer` | The residual analysis component of the hybrid detector. |

