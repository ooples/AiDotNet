---
title: "DYNOTEARSAlgorithm<T>"
description: "DYNOTEARS — Dynamic NOTEARS for time series structure learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

DYNOTEARS — Dynamic NOTEARS for time series structure learning.

## For Beginners

DYNOTEARS is like NOTEARS but for time series. It can learn
both "X and Y affect each other at the same time" and "yesterday's X affects today's Y"
type relationships simultaneously, using the same elegant continuous optimization approach.
The key insight is that only the contemporaneous matrix W needs to be acyclic — lagged
effects can't create instantaneous cycles.

## How It Works

DYNOTEARS extends the NOTEARS continuous optimization framework to time series data.
It jointly learns both contemporaneous (W) and lagged (A₁, ..., Aₖ) adjacency matrices
using an augmented Lagrangian with the acyclicity constraint only on the contemporaneous matrix W.

**Model:** X(t) = W^T X(t) + Σ_k A_k^T X(t-k) + e(t)
**Objective:** min_{W,A} ½n⁻¹ ||X_t - X_t W - Z A||²_F + λ₁(||W||₁ + ||A||₁)
**Constraint:** h(W) = tr(e^(W∘W)) - d = 0 (acyclicity only on contemporaneous W)

Reference: Pamfil et al. (2020), "DYNOTEARS: Structure Learning from Time-Series Data", AISTATS.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAcyclicity(Matrix<>,Int32)` | Computes h(W) = tr(e^{W∘W}) - d, the NOTEARS acyclicity constraint. |
| `ComputeMatrixExponentialOfHadamard(Matrix<>,Int32)` | Computes e^{W∘W} via Taylor series (needed for gradient of h(W)). |
| `DiscoverStructureCore(Matrix<>)` |  |
| `OptimizeInner(Matrix<>,Matrix<>,Matrix<>,Matrix<>,Int32,Int32,Int32,Double,Double)` | Inner optimization loop: gradient descent on the joint (W, A) objective. |

