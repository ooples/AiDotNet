---
title: "FinancialSearchSpace"
description: "Provides default AutoML search spaces for finance models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.AutoML`

Provides default AutoML search spaces for finance models.

## For Beginners

AutoML can either just pick a model or also tune settings.
This class tells AutoML which settings it is allowed to tune and what ranges
are reasonable for each setting. For example, a learning rate should typically
be between 0.0001 and 0.1, while the number of layers might be 1 to 8.

## How It Works

The search space defines which hyperparameters AutoML is allowed to explore.
Each model type has specific hyperparameters that can be tuned during AutoML
optimization, such as learning rate, hidden size, and dropout rate.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialSearchSpace(FinancialDomain)` | Initializes a new search space provider for the chosen finance domain. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetCommonForecastingSearchSpace` | Gets common search space parameters for forecasting models. |
| `GetCommonRiskSearchSpace` | Gets common search space parameters for risk models. |
| `GetCommonTabularSearchSpace` | Gets common search space parameters for tabular models. |
| `GetDeepARSearchSpace` | Gets the search space for DeepAR model. |
| `GetDefaultSearchSpace` | Gets a default search space for unsupported model types. |
| `GetDomainAwareDefaultSearchSpace` | Gets a domain-aware default search space when no specific model is requested. |
| `GetITransformerSearchSpace` | Gets the search space for iTransformer model. |
| `GetNBEATSSearchSpace` | Gets the search space for N-BEATS model. |
| `GetNeuralVaRSearchSpace` | Gets the search space for NeuralVaR model. |
| `GetPatchTSTSearchSpace` | Gets the search space for PatchTST model. |
| `GetSearchSpace(Type)` | Gets the default search space for a specific model type. |
| `GetTFTSearchSpace` | Gets the search space for Temporal Fusion Transformer (TFT) model. |
| `GetTabNetSearchSpace` | Gets the search space for TabNet model. |
| `GetTabTransformerSearchSpace` | Gets the search space for TabTransformer model. |

