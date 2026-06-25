---
title: "IBiasDetector<T>"
description: "Defines an interface for detecting bias in machine learning model predictions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for detecting bias in machine learning model predictions.

## How It Works

**For Beginners:** This interface helps identify unfair treatment in machine learning models.

In machine learning, bias occurs when a model treats different groups of people unfairly.
For example, a loan approval model might unfairly reject applications from certain demographic groups.

This interface provides methods to:

- Detect when a model is biased against certain groups
- Measure how severe the bias is
- Compare different models to find which one is most fair

The bias score measures how unfair the model is. Important points:

- Lower bias scores usually indicate fairer models
- The score helps us identify which groups are being treated unfairly
- We can use these measurements to improve our models and make them more equitable

## Properties

| Property | Summary |
|:-----|:--------|
| `IsLowerBiasBetter` | Indicates whether lower bias scores represent better (fairer) models. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectBias(Vector<>,Vector<>,Vector<>)` | Detects bias in model predictions by analyzing predictions across different groups. |
| `IsBetterBiasScore(,)` | Compares two bias scores and determines if the current score represents better (fairer) performance. |

