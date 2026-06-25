---
title: "TiltedERMFairness<T>"
description: "Implements TERM (Tilted Empirical Risk Minimization) for fairness in FL."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Implements TERM (Tilted Empirical Risk Minimization) for fairness in FL.

## For Beginners

TERM smoothly interpolates between average and worst-case
optimization using a tilt parameter t. When t=0, it's standard average loss. When t>0,
it up-weights high-loss clients (moving toward worst-case). When t<0, it focuses on
easy clients (useful for outlier robustness). This gives a smooth, differentiable fairness
objective that's easier to optimize than the hard minimax of AFL.

## How It Works

Objective:

Reference: Li, T., et al. (2021). "Tilted Empirical Risk Minimization." ICLR 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TiltedERMFairness(Double)` | Creates a new TERM fairness instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Tilt` | Gets the tilt parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeObjective(Dictionary<Int32,Double>)` | Computes the TERM objective value. |
| `ComputeWeights(Dictionary<Int32,Double>)` | Computes TERM aggregation weights based on client losses. |

