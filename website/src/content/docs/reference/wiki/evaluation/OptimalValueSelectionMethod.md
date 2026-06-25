---
title: "OptimalValueSelectionMethod"
description: "Method for selecting optimal hyperparameter value from validation curve."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Options`

Method for selecting optimal hyperparameter value from validation curve.

## Fields

| Field | Summary |
|:-----|:--------|
| `ElbowMethod` | Elbow method: find point where improvement diminishes. |
| `MaxValidationScore` | Select value with highest validation score. |
| `MinTrainTestGap` | Select value where train and validation scores are closest. |
| `OneSERule` | One standard error rule: select simplest value within 1 SE of maximum. |

