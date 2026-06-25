---
title: "AutoMLOptions<T, TInput, TOutput>"
description: "Configuration options for running AutoML through the AiDotNet facade."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for running AutoML through the AiDotNet facade.

## For Beginners

AutoML is an automatic "model picker + tuner".
You choose a budget, and AutoML tries different models/settings to find a strong performer.

## How It Works

This options class is designed for use with `AiModelBuilder`.
It follows the AiDotNet facade pattern: users provide minimal configuration, and the library supplies
industry-standard defaults internally.

## Properties

| Property | Summary |
|:-----|:--------|
| `Budget` | Gets or sets the compute budget for the AutoML run. |
| `CrossValidation` | Gets or sets cross-validation options for trial evaluation. |
| `Ensembling` | Gets or sets optional ensembling options applied after the AutoML search completes. |
| `MultiFidelity` | Gets or sets multi-fidelity options used when `SearchStrategy` is `MultiFidelity`. |
| `NAS` | Gets or sets Neural Architecture Search (NAS) specific AutoML options. |
| `OptimizationMetricOverride` | Gets or sets an optional optimization metric override. |
| `ReinforcementLearning` | Gets or sets reinforcement-learning specific AutoML options. |
| `SearchStrategy` | Gets or sets the search strategy used for hyperparameter/model exploration. |
| `TaskFamilyOverride` | Gets or sets an optional task family override. |

