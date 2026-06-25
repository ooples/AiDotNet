---
title: "NeuralNetworkDerivatives<T>"
description: "Provides first- and second-order derivatives for neural networks with safe fallbacks."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Autodiff`

Provides first- and second-order derivatives for neural networks with safe fallbacks.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDerivatives(NeuralNetworkBase<>,[],Int32)` | Computes first and second derivatives for a feedforward network at a single input point. |
| `ComputeDerivativesSPSA(NeuralNetworkBase<>,[],Int32,Double,Int32)` | SPSA-based derivative estimation for high-dimensional inputs. |
| `ComputeHessian(NeuralNetworkBase<>,[],Int32)` | Computes the Hessian for a scalar output index. |

## Fields

| Field | Summary |
|:-----|:--------|
| `SPSAThreshold` | Threshold above which SPSA is used instead of per-dimension finite differences. |

