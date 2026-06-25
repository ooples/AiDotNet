---
title: "EnsembleFitDetector<T, TInput, TOutput>"
description: "A fit detector that combines the results of multiple individual fit detectors to provide a more robust assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that combines the results of multiple individual fit detectors to provide a more robust assessment.

## For Beginners

An ensemble approach combines the opinions of multiple "experts" (in this case, 
different fit detectors) to make a more reliable decision. This is similar to getting second and third 
opinions from different doctors before making an important medical decision.

## How It Works

This detector aggregates the results from multiple fit detectors, potentially giving different weights 
to each detector's opinion, to determine the overall fit type, confidence level, and recommendations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnsembleFitDetector(List<IFitDetector<,,>>,EnsembleFitDetectorOptions)` | Initializes a new instance of the EnsembleFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level by combining the confidence levels of all individual detectors. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model by combining the results of multiple individual fit detectors. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type by combining the assessments of all individual detectors. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations by combining the recommendations of all individual detectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_detectors` | The list of individual fit detectors that make up the ensemble. |
| `_options` | Configuration options for the ensemble fit detector. |

