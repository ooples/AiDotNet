---
title: "VisionLanguageActionOptions"
description: "Base configuration options for Vision-Language-Action (VLA) models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Robotics`

Base configuration options for Vision-Language-Action (VLA) models.

## For Beginners

These options configure the VisionLanguageAction model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VisionLanguageActionOptions` | Initializes a new instance with default values. |
| `VisionLanguageActionOptions(VisionLanguageActionOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionDimension` | Gets or sets the action space dimensionality (e.g., number of joint DOFs). |
| `LanguageModelName` | Gets or sets the language model backbone name. |
| `ObservationHistory` | Gets or sets the observation history length (number of past frames). |
| `PredictionHorizon` | Gets or sets the maximum action prediction horizon (number of future steps). |

