---
title: "IRiskModel<T>"
description: "Interface for financial risk models that estimate potential losses."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

Interface for financial risk models that estimate potential losses.

## For Beginners

Risk models help answer: "How much could I lose?"

**Key Risk Measures:**

- **Value at Risk (VaR):** Maximum loss at a given confidence level
- **Conditional VaR (CVaR):** Average loss when VaR is exceeded
- **Expected Shortfall:** Same as CVaR, also called tail risk

**Example:**
If 95% VaR is $1M, there's a 95% chance you won't lose more than $1M today.
The 5% CVaR tells you the average loss in the worst 5% of cases.

**Why Use Neural Risk Models:**

- Capture non-linear risk patterns
- Handle complex portfolio structures
- Adapt to changing market conditions
- Process high-dimensional data

## How It Works

Risk models are essential for financial risk management, providing estimates of potential
losses under various market conditions. This interface defines common methods for risk
measurement, stress testing, and risk decomposition.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceLevel` | Gets the confidence level for risk calculations (e.g., 0.95 for 95% VaR). |
| `TimeHorizon` | Gets the time horizon for risk calculations in days. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCVaR(Tensor<>,Tensor<>)` | Calculates Conditional Value at Risk (CVaR), also known as Expected Shortfall. |
| `CalculateVaR(Tensor<>,Tensor<>)` | Calculates Value at Risk (VaR) for the given portfolio. |
| `DecomposeRisk(Tensor<>,Tensor<>)` | Decomposes total risk into component contributions. |
| `EstimateExceedanceProbability(Tensor<>,Tensor<>,)` | Estimates the probability of a given loss being exceeded. |
| `GetRiskMetrics` | Gets risk-specific metrics for model evaluation. |
| `StressTest(Tensor<>,Tensor<>)` | Performs stress testing under specified scenarios. |

