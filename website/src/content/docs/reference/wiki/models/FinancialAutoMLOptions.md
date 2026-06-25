---
title: "FinancialAutoMLOptions<T>"
description: "Configuration options for financial AutoML runs."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for financial AutoML runs.

## For Beginners

Use this class to say "I want AutoML for forecasting" or
"I want AutoML for risk." Then provide your architecture and budget.

## How It Works

FinancialAutoML selects among finance-specific models based on the chosen domain.
It follows the facade pattern: you provide a minimal setup and the library supplies defaults.

## Properties

| Property | Summary |
|:-----|:--------|
| `Architecture` | Gets or sets the user-provided neural network architecture. |
| `Budget` | Gets or sets the compute budget for the AutoML run. |
| `CandidateModels` | Gets or sets an optional list of candidate models to consider. |
| `CrossValidation` | Gets or sets optional cross-validation settings. |
| `Domain` | Gets or sets the finance domain to search. |
| `OptimizationMetricOverride` | Gets or sets an optional optimization metric override. |
| `SearchStrategy` | Gets or sets the AutoML search strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and throws if required values are missing. |

