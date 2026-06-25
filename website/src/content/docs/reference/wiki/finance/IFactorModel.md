---
title: "IFactorModel<T>"
description: "Interface for financial factor models that learn latent factors from market data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

Interface for financial factor models that learn latent factors from market data.

## For Beginners

Factor models help explain why asset prices move together.

**The Key Insight:**
Asset returns can be broken down into:

- **Factor returns:** Movements explained by common factors (market, size, value, momentum, etc.)
- **Idiosyncratic returns:** Stock-specific movements not explained by factors

For example, if tech stocks all go up together, that's likely a "tech sector factor."
If just Apple goes up because of a product launch, that's idiosyncratic to Apple.

**Why Use Factor Models:**

- Risk decomposition: Understand what risks drive your portfolio
- Alpha generation: Find factors that predict future returns
- Portfolio construction: Build portfolios with desired factor exposures
- Risk management: Hedge specific factor risks

**Common Factor Categories:**

- Style factors: Value, growth, momentum, quality, volatility
- Macro factors: Interest rates, inflation, GDP growth
- Statistical factors: PCA-derived factors, machine-learned factors
- Fundamental factors: Earnings, book value, leverage

## How It Works

Factor models decompose asset returns into systematic factors and idiosyncratic noise.
This interface extends `IFinancialModel` with factor-specific capabilities
for extracting, analyzing, and using latent factors in quantitative finance.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumAssets` | Gets the number of assets (securities) the model can handle. |
| `NumFactors` | Gets the number of latent factors the model learns or uses. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAlpha(Tensor<>,Tensor<>)` | Computes alpha (expected excess return) for each asset. |
| `ExtractFactors(Tensor<>)` | Extracts latent factors from market data. |
| `GetFactorCovariance(Tensor<>)` | Computes the factor covariance matrix. |
| `GetFactorLoadings(Tensor<>)` | Computes factor loadings (exposures) for each asset. |
| `GetFactorMetrics` | Gets factor-specific performance metrics. |
| `PredictReturns(Tensor<>)` | Predicts expected returns based on factor exposures. |

