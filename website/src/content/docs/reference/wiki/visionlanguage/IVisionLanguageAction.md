---
title: "IVisionLanguageAction<T>"
description: "Interface for Vision-Language-Action (VLA) models that connect visual understanding and language reasoning to physical robotic actions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for Vision-Language-Action (VLA) models that connect visual understanding
and language reasoning to physical robotic actions.

## How It Works

VLA models bridge the gap between perception (vision-language understanding) and action
(robotic control). They take visual observations and optional language instructions to
predict action sequences for robot manipulation, navigation, and planning.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionDimension` | Gets the dimensionality of the action space (e.g., number of joint DOFs). |
| `LanguageModelName` | Gets the name of the language model backbone. |

## Methods

| Method | Summary |
|:-----|:--------|
| `PredictAction(Tensor<>,String)` | Predicts an action sequence from a visual observation and a language instruction. |

