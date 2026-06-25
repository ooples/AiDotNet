---
title: "AutoMLBudgetOptions"
description: "Configuration options that control AutoML compute budgets."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options that control AutoML compute budgets.

## For Beginners

This controls how much time AutoML is allowed to spend searching for a good model.
If you want faster results, choose a smaller preset or set a shorter time limit / fewer trials.

## How It Works

AutoML budgets define how long AutoML is allowed to search and how many trials it can run.

## Properties

| Property | Summary |
|:-----|:--------|
| `Preset` | Gets or sets the budget preset used when explicit overrides are not provided. |
| `TimeLimitOverride` | Gets or sets an optional time limit override for the AutoML run. |
| `TrialLimitOverride` | Gets or sets an optional trial limit override for the AutoML run. |

