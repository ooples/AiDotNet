---
title: "TransferEntropyAlgorithm<T>"
description: "Transfer Entropy — information-theoretic measure of directed information flow."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.InformationTheoretic`

Transfer Entropy — information-theoretic measure of directed information flow.

## For Beginners

Transfer Entropy is like Granger causality but works for nonlinear
relationships too. It asks: "Does knowing X's past reduce my uncertainty about Y's future,
beyond what Y's own past already tells me?" If yes, X transfers information to Y.

## How It Works

Transfer entropy quantifies the amount of directed information transfer from one
process to another. It measures the reduction in uncertainty of Y's future given
the past of both X and Y, compared to only Y's past. It is a nonlinear generalization
of Granger causality.

**Algorithm:**

- For each pair (X→Y), construct lagged variables: Y_past (lag 1..L of Y), X_past (lag 1..L of X)
- Compute TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
- Using Gaussian approximation: TE = 0.5 * log(var(Y_future|Y_past) / var(Y_future|Y_past,X_past))
- Apply score threshold: only keep edges where TE exceeds threshold
- Direction is inherent: TE(X→Y) ≠ TE(Y→X) in general

Reference: Schreiber (2000), "Measuring Information Transfer", Physical Review Letters.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |
| `SupportsTimeSeries` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCorrelationFallback(Matrix<>,Int32,Int32,Int32)` | Fallback for deterministic data where OLS residuals are near-zero. |
| `DiscoverStructureCore(Matrix<>)` |  |

