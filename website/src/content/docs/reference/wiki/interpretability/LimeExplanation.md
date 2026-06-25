---
title: "LimeExplanation<T>"
description: "Represents a LIME (Local Interpretable Model-agnostic Explanations) explanation for a prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability`

Represents a LIME (Local Interpretable Model-agnostic Explanations) explanation for a prediction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LimeExplanation` | Initializes a new instance of the LimeExplanation class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureImportance` | Gets or sets the feature importance scores for the explanation. |
| `Intercept` | Gets or sets the intercept of the linear approximation. |
| `LocalModelScore` | Gets or sets the R-squared score of the local linear approximation. |
| `NumFeatures` | Gets or sets the number of features used in the explanation. |
| `PredictedValue` | Gets or sets the predicted value for the explained instance. |

