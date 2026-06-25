---
title: "AdaptiveRandomizedSmoothing<T, TInput, TOutput>"
description: "Implements Adaptive Randomized Smoothing with f-Differential Privacy (f-DP) certified defense."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Defenses`

Implements Adaptive Randomized Smoothing with f-Differential Privacy (f-DP) certified defense.

## For Beginners

Standard Randomized Smoothing adds the same amount of noise to every
input. But some inputs are "easy" (far from the decision boundary) and don't need much noise,
while "hard" inputs (near the boundary) need more. This adaptive version adjusts the noise
level for each input — like adjusting your seatbelt based on road conditions rather than
always wearing the tightest setting.

## How It Works

Extends standard Randomized Smoothing by adapting the noise distribution per-input based
on a local sensitivity estimate. Instead of using a fixed Gaussian sigma, the noise level
is scaled according to the model's local Lipschitz constant around each input. This provides
tighter certified radii for "easy" inputs while maintaining coverage for difficult ones.

**References:**

- Adaptive Randomized Smoothing: f-DP certified defense (NeurIPS 2024)
- Certified Adversarial Robustness via Randomized Smoothing (Cohen et al., 2019)
- Tight second-order certificates for randomized smoothing (Mohapatra et al., 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaptiveRandomizedSmoothing(CertifiedDefenseOptions<>,Double,Double,Int32)` | Initializes a new instance of Adaptive Randomized Smoothing. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CertifyBatch([],IFullModel<,,>)` |  |
| `CertifyPrediction(,IFullModel<,,>)` |  |
| `ComputeCertifiedRadius(,IFullModel<,,>)` |  |
| `Deserialize(Byte[])` |  |
| `EvaluateCertifiedAccuracy([],[],IFullModel<,,>,)` |  |
| `GetOptions` |  |
| `LoadModel(String)` |  |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |

