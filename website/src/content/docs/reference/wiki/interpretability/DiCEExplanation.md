---
title: "DiCEExplanation<T>"
description: "DiCE explanation containing multiple diverse counterfactuals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

DiCE explanation containing multiple diverse counterfactuals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiCEExplanation(Vector<>,,,List<SingleCounterfactual<>>,String[])` | Initializes a new DiCE explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageDistance` | Average distance from original across counterfactuals. |
| `AverageSparsity` | Average number of features changed across counterfactuals. |
| `Count` | Number of counterfactuals generated. |
| `Counterfactuals` | Diverse counterfactual explanations. |
| `FeatureNames` | Feature names. |
| `OriginalInstance` | Original input instance. |
| `OriginalPrediction` | Original prediction. |
| `TargetPrediction` | Target prediction value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetClosest` | Gets the counterfactual closest to original. |
| `GetCommonChanges` | Gets features that appear in multiple counterfactuals. |
| `GetSparsest` | Gets the counterfactual with fewest changes. |
| `ToString` | Returns string representation. |

