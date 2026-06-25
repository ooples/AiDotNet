---
title: "SceneClassificationResult<T>"
description: "Result of scene classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of scene classification.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllProbabilities` | Gets all probabilities as a dictionary (legacy API compatibility). |
| `AllScenes` | Gets or sets all scene predictions sorted by probability. |
| `Category` | Gets or sets the scene category (indoor/outdoor/transportation). |
| `Characteristics` | Gets or sets detected acoustic characteristics. |
| `Confidence` | Gets or sets the confidence score. |
| `Features` | Gets or sets extracted features used for classification (legacy API compatibility). |
| `PredictedScene` | Gets or sets the predicted scene. |
| `TopPredictions` | Gets top predictions as a list of tuples (legacy API compatibility). |

