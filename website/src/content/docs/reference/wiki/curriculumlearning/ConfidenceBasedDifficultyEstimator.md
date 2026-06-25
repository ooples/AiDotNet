---
title: "ConfidenceBasedDifficultyEstimator<T, TInput, TOutput>"
description: "Difficulty estimator based on model prediction confidence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.DifficultyEstimators`

Difficulty estimator based on model prediction confidence.

## For Beginners

This estimator uses the model's confidence in its
predictions as a measure of difficulty. Low confidence predictions indicate
the model is uncertain, suggesting the sample is harder.

## How It Works

**How It Works:**

**Confidence Metrics:**

**References:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConfidenceBasedDifficultyEstimator(ConfidenceMetricType,Boolean)` | Initializes a new instance of the `ConfidenceBasedDifficultyEstimator` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this estimator. |
| `RequiresModel` | Gets whether this estimator requires the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySoftmax(Vector<>)` | Applies softmax to convert logits to probabilities. |
| `CalculateConfidence(Vector<>)` | Calculates confidence from probabilities based on the selected metric. |
| `CalculateEntropyConfidence(Vector<>)` | Calculates confidence from entropy (low entropy = high confidence). |
| `CalculateMarginConfidence(Vector<>)` | Calculates confidence as margin between top two probabilities. |
| `CalculateMaxProbabilityConfidence(Vector<>)` | Calculates confidence as maximum probability. |
| `ClampProbability()` | Clamps a value to valid probability range [0, 1]. |
| `ConvertToProbabilities()` | Converts a prediction output to a probability vector. |
| `EstimateDifficulties(IDataset<,,>,IFullModel<,,>)` | Estimates difficulty scores for all samples. |
| `EstimateDifficulty(,,IFullModel<,,>)` | Estimates the difficulty of a single sample based on prediction confidence. |
| `EstimateDifficultyWithConfidence(,,IFullModel<,,>)` | Gets both difficulty estimate and confidence for a sample. |
| `GetPredictionProbabilities(IFullModel<,,>,)` | Gets prediction probabilities from the model. |

