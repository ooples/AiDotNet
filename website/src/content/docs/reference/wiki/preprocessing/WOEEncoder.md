---
title: "WOEEncoder<T>"
description: "Encodes categorical features using Weight of Evidence (WOE)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using Weight of Evidence (WOE).

## For Beginners

WOE tells you how "good" or "bad" a category is for prediction:

- WOE > 0: Category is more likely to have positive outcomes
- WOE < 0: Category is more likely to have negative outcomes
- WOE ≈ 0: Category has no predictive power

Example in loan default prediction:

- "Employed" might have WOE = -0.5 (less likely to default)
- "Unemployed" might have WOE = +0.8 (more likely to default)

## How It Works

Weight of Evidence is commonly used in credit scoring and binary classification.
It measures the strength of the relationship between a category and the binary target.
WOE = ln(Distribution of Events / Distribution of Non-Events)

Higher WOE values indicate categories more associated with the positive class,
while lower (negative) values indicate association with the negative class.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WOEEncoder(Double,WOEHandleUnknown,Int32[])` | Creates a new instance of `WOEEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HandleUnknown` | Gets how unknown categories are handled. |
| `Regularization` | Gets the regularization parameter to prevent infinite WOE values. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `WOEValues` | Gets the WOE values for each category. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateInformationValue(Matrix<>,Vector<>)` | Calculates Information Value (IV) for each feature. |
| `Fit(Matrix<>,Vector<>)` | Fits the encoder by computing WOE values for each category. |
| `FitCore(Matrix<>)` | Fits the encoder (requires binary target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by replacing categories with WOE values. |

