---
title: "BuiltInSupervisedAutoMLModelBase<T, TInput, TOutput>"
description: "Base class for built-in supervised AutoML strategies that operate on tabular Matrix/Vector tasks."
section: "API Reference"
---

`Base Classes` · `AiDotNet.AutoML`

Base class for built-in supervised AutoML strategies that operate on tabular Matrix/Vector tasks.

## For Beginners

AutoML tries many "trials". Each trial picks a model family and some settings, then trains
and scores it. Different strategies decide which settings to try next.

## How It Works

This class centralizes built-in model construction and default candidate selection so different search
strategies (random, Bayesian, evolutionary, etc.) can focus on "how to propose the next trial".

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsureDefaultCandidateModels(,)` | Applies built-in default candidate models when the user has not configured candidates explicitly. |

