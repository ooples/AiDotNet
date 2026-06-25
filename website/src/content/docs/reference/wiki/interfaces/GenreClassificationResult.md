---
title: "GenreClassificationResult<T>"
description: "Result of genre classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of genre classification.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllGenres` | Gets or sets all predicted genres (for multi-label classification). |
| `AllProbabilities` | Gets all probabilities as a dictionary (legacy API compatibility). |
| `Confidence` | Gets or sets the confidence for the predicted genre. |
| `Features` | Gets or sets extracted features used for classification (legacy API compatibility). |
| `IsMultiLabel` | Gets or sets whether this is a multi-label result. |
| `PredictedGenre` | Gets or sets the predicted genre (or primary genre if multi-label). |
| `TopPredictions` | Gets top predictions as a list of tuples (legacy API compatibility). |

