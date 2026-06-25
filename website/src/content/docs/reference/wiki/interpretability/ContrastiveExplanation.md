---
title: "ContrastiveExplanation<T>"
description: "Represents the result of a Contrastive explanation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a Contrastive explanation.

## Properties

| Property | Summary |
|:-----|:--------|
| `FactClass` | Gets or sets the fact class (what was predicted). |
| `FactClassName` | Gets or sets the name of the fact class. |
| `FactScore` | Gets or sets the score for the fact class. |
| `FeatureContributions` | Gets or sets feature contributions to the fact-vs-foil decision. |
| `FeatureNames` | Gets or sets the feature names. |
| `FoilClass` | Gets or sets the foil class (the alternative). |
| `FoilClassName` | Gets or sets the name of the foil class. |
| `FoilScore` | Gets or sets the score for the foil class. |
| `Input` | Gets or sets the input instance. |
| `PertinentNegatives` | Gets or sets the pertinent negatives (features that could flip to foil). |
| `PertinentPositives` | Gets or sets the pertinent positives (features supporting the fact). |
| `Prediction` | Gets or sets the model prediction. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a human-readable summary. |

