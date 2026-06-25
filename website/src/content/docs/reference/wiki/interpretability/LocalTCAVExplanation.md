---
title: "LocalTCAVExplanation<T>"
description: "Represents a local TCAV explanation for a single input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents a local TCAV explanation for a single input.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LocalTCAVExplanation(Vector<>,String,Int32,,,,String)` | Initializes a new local TCAV explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConceptName` | Gets the concept name. |
| `ConceptProjection` | Gets the projection of activations onto the concept direction. |
| `DirectionalDerivative` | Gets the directional derivative (concept sensitivity). |
| `InfluenceDirection` | Gets the influence direction ("positive" or "negative"). |
| `Input` | Gets the input that was explained. |
| `Prediction` | Gets the prediction score for the target class. |
| `TargetClass` | Gets the target class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a human-readable summary. |

