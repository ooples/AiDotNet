---
title: "TiMINoAlgorithm<T>"
description: "TiMINo — Time series Models with Independent Noise."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

TiMINo — Time series Models with Independent Noise.

## For Beginners

TiMINo checks if the "leftover noise" after predicting one variable
from another's past is truly random (independent). If it is, the prediction direction is
likely the causal direction.

## How It Works

TiMINo tests whether the residuals of a time series regression model are independent
of the inputs. For each pair (i→j), it fits a linear model predicting j from lagged
values of i (and j itself), then tests whether the residuals are independent of i's
lagged values using the HSIC independence criterion.

**Algorithm:**

- For each pair (i,j), fit model: x_j[t] = sum_l (a_l * x_i[t-l] + b_l * x_j[t-l]) + noise
- Compute residuals: e[t] = x_j[t] - predicted
- Test independence of residuals from input using HSIC
- If residuals are independent of i's lags, direction i→j is valid
- Compare HSIC scores for i→j vs j→i to determine direction
- Edge weight from OLS coefficient magnitude

Reference: Peters et al. (2013), "Causal Discovery with Continuous Additive Noise Models", JMLR.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CorrelationDenominatorEpsilon` | Epsilon for preventing division by zero in correlation denominators. |
| `EdgeCoefficientThreshold` | Minimum absolute standardized coefficient magnitude for an edge to be included. |
| `IndependenceThreshold` | Squared correlation threshold below which residuals are considered independent of the cause. |
| `NumericalStabilityEpsilon` | Epsilon for numerical stability in variance/correlation computations. |

