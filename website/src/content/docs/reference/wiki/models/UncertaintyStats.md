---
title: "UncertaintyStats<T>"
description: "Represents uncertainty-quantification diagnostics aggregated over a dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents uncertainty-quantification diagnostics aggregated over a dataset.

## For Beginners

This stores summary uncertainty metrics (like average entropy) for an entire dataset,
similar to how accuracy or error metrics summarize model quality.

## How It Works

This container is designed to integrate with the existing AiDotNet evaluation pipeline by living alongside
`ErrorStats` and `PredictionStats` within `DataSetStats`.

## Properties

| Property | Summary |
|:-----|:--------|
| `Metrics` | Gets a dictionary of aggregate uncertainty metrics for the dataset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Empty` | Creates an empty `UncertaintyStats` instance. |

