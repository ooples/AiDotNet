---
title: "DINOConfig"
description: "DINO-specific configuration settings."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SelfSupervisedLearning`

DINO-specific configuration settings.

## For Beginners

DINO (self-DIstillation with NO labels) uses self-distillation
with centering and sharpening to learn emergent attention patterns in Vision Transformers.

## Properties

| Property | Summary |
|:-----|:--------|
| `CenterMomentum` | Gets or sets the momentum for the centering mechanism. |
| `NumGlobalCrops` | Gets or sets the number of global crops (large crops). |
| `NumLocalCrops` | Gets or sets the number of local crops (small crops). |
| `StudentTemperature` | Gets or sets the student temperature. |
| `TeacherTemperatureEnd` | Gets or sets the final teacher temperature. |
| `TeacherTemperatureStart` | Gets or sets the initial teacher temperature. |
| `TeacherTemperatureWarmupEpochs` | Gets or sets the number of warmup epochs for teacher temperature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

