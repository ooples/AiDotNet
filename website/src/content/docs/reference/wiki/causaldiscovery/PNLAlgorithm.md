---
title: "PNLAlgorithm<T>"
description: "PNL (Post-Nonlinear Causal Model) — Y = g(f(X) + N)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Functional`

PNL (Post-Nonlinear Causal Model) — Y = g(f(X) + N).

## For Beginners

Sometimes the relationship between cause and effect has two layers
of nonlinearity: first the cause produces an effect through some function f, then
that result gets distorted by another function g (like a sensor with nonlinear response).
PNL can handle this double-nonlinearity by "undoing" the outer distortion before
testing for the cause-effect direction.

## How It Works

PNL extends the Additive Noise Model by allowing a post-nonlinear distortion g.
The model is Y = g(f(X) + N) where f is the causal mechanism, N is independent noise,
and g is an invertible post-nonlinear transformation. The algorithm:

- For each variable pair, fits the inner function f via kernel regression.
- Estimates the post-nonlinear function g by applying a rank-based CDF transform

(probability integral transform) to approximate g^(-1).

- Computes residuals in the "linearized" space after inverting g.
- Tests independence of the residuals from the cause using mutual information.
- Orients the edge in the direction where residuals are more independent.

Reference: Zhang and Hyvarinen (2009), "On the Identifiability of the
Post-Nonlinear Causal Model", UAI.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `PNLResidualDependence(Vector<>,Vector<>)` | Computes residual dependence for the PNL model Y = g(f(X) + N). |
| `ProbitApproximation(Double)` | Abramowitz and Stegun 26.2.23 rational approximation of the probit function. |
| `RankTransform(Vector<>)` | Rank-based transform: maps values to approximate standard normal via empirical CDF. |

