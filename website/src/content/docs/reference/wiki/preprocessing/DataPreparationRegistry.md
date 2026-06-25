---
title: "DataPreparationRegistry<T>"
description: "Global registry for the data preparation pipeline."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Preprocessing.DataPreparation`

Global registry for the data preparation pipeline.

## For Beginners

This is an internal component that stores your data preparation settings.
You don't need to interact with this directly - just use AiModelBuilder:

The configured data preparation is automatically applied during training.

## How It Works

DataPreparationRegistry provides a singleton pattern for managing the active data preparation
pipeline. This handles row-changing operations like outlier removal and data augmentation.

## Properties

| Property | Summary |
|:-----|:--------|
| `Current` | Gets or sets the current data preparation pipeline. |
| `IsConfigured` | Gets whether a data preparation pipeline is currently configured. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears the current data preparation pipeline. |
| `FitResample(Matrix<>,Vector<>)` | Applies the current data preparation pipeline to data. |

