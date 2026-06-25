---
title: "BiasDetectorBase<T>"
description: "Base class for all bias detectors that identify unfair treatment in model predictions."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Interpretability`

Base class for all bias detectors that identify unfair treatment in model predictions.

## For Beginners

This is a foundation class that all bias detectors build upon.

Think of a bias detector like an inspector checking for fairness:

- It examines how your model makes predictions for different groups of people
- It identifies when certain groups are being treated unfairly
- It provides metrics that measure the severity of the bias

Different bias detectors might look for different types of unfairness, but they all
share common functionality. This base class provides that shared foundation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiasDetectorBase(Boolean)` | Initializes a new instance of the BiasDetectorBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsLowerBiasBetter` | Gets a value indicating whether lower bias scores represent better (fairer) models. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectBias(Vector<>,Vector<>,Vector<>)` | Detects bias in model predictions by analyzing predictions across different groups. |
| `GetBiasDetectionResult(Vector<>,Vector<>,Vector<>)` | Abstract method that must be implemented by derived classes to perform specific bias detection logic. |
| `IsBetterBiasScore(,)` | Determines whether a new bias score represents better (fairer) performance than the current best score. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_isLowerBiasBetter` | Indicates whether lower bias scores represent better (fairer) models. |
| `_numOps` | Provides mathematical operations for the specific numeric type being used. |

