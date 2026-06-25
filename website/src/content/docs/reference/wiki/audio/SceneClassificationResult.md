---
title: "SceneClassificationResult"
description: "Result of acoustic scene classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

Result of acoustic scene classification.

## For Beginners

SceneClassificationResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllProbabilities` | Probabilities for all scenes. |
| `Category` | Scene category (indoor, outdoor_urban, outdoor_nature, transportation). |
| `Confidence` | Confidence score for predicted scene (0-1). |
| `Features` | Extracted features used for classification. |
| `PredictedScene` | Most likely scene. |
| `TopPredictions` | Top K predictions with probabilities. |

