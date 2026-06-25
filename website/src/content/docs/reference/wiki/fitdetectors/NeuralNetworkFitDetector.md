---
title: "NeuralNetworkFitDetector<T, TInput, TOutput>"
description: "A specialized detector for evaluating the fit quality of neural network models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A specialized detector for evaluating the fit quality of neural network models.

## For Beginners

This class helps you understand if your neural network is performing well or not.
It analyzes how your model performs on different data sets and gives you recommendations
on how to improve it.

Think of it like a health check for your neural network that tells you:

- If your model is working well (good fit)
- If it's memorizing the training data instead of learning patterns (overfitting)
- If it's not complex enough to learn the patterns in your data (underfitting)
- What steps you can take to improve your model

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralNetworkFitDetector(NeuralNetworkFitDetectorOptions)` | Creates a new instance of the neural network fit detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `_overfittingScore` | A measure of how much the model is overfitting to the training data. |
| `_testLoss` | The error measurement on the test dataset. |
| `_trainingLoss` | The error measurement on the training dataset. |
| `_validationLoss` | The error measurement on the validation dataset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates how confident the detector is in its fit assessment. |
| `CalculateOverfittingScore(ModelEvaluationData<,,>)` | Calculates a score that measures how much the model is overfitting to the training data. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes the model's performance data and determines the quality of fit. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on validation loss and overfitting score. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations for improving the model based on its fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the neural network fit detector. |

