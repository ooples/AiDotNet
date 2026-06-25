---
title: "IAutoMLAugmentation<T, TData>"
description: "Interface for augmentations that expose their hyperparameter search space."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation`

Interface for augmentations that expose their hyperparameter search space.

## For Beginners

AutoML systems can automatically find the best
augmentation settings (like rotation angle ranges or color adjustment strength)
by searching through possible configurations and measuring their effect on
model performance.

## How It Works

This interface enables AutoML systems to automatically tune augmentation
hyperparameters during neural architecture search or hyperparameter optimization.

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateWithParameters(IDictionary<String,Object>)` | Creates a new instance with the specified hyperparameters. |
| `GetSearchSpace` | Gets the hyperparameter search space for this augmentation. |
| `ValidateParameters(IDictionary<String,Object>)` | Validates hyperparameter values. |

