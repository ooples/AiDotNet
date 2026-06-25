---
title: "FedSamAggregationStrategy<T>"
description: "Implements FedSAM (Sharpness-Aware Minimization for Federated Learning) aggregation strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements FedSAM (Sharpness-Aware Minimization for Federated Learning) aggregation strategy.

## For Beginners

Regular optimization finds a low point (minimum) in the loss
landscape, but this point might be "sharp" — a small change in parameters causes a large
change in loss. FedSAM instead seeks "flat" minima that are more robust, which is especially
important in FL where each client's data creates a different loss landscape.

## How It Works

Local training uses a two-step process per batch:

Variants:

Reference: Caldarola, D., et al. (2022). "Improving Generalization in Federated Learning
by Seeking Flat Minima."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedSamAggregationStrategy(Double,FedSamVariant,Double,Double,Double)` | Initializes a new instance of the `FedSamAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ApproximationCoefficient` | Gets the approximation coefficient (FedSpeed/FedLESAM). |
| `ControlCoefficient` | Gets the stochastic control coefficient (FedSCAM). |
| `GlobalPerturbationRadius` | Gets the global perturbation radius (FedSMOO). |
| `PerturbationRadius` | Gets the SAM perturbation radius (rho). |
| `Variant` | Gets the FedSAM variant being used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates client models using standard weighted averaging. |
| `ApplyPerturbation(Dictionary<String,[]>,Dictionary<String,[]>)` | Applies a perturbation to model parameters: w_perturbed = w + epsilon. |
| `ComputeLESAMPerturbation(Dictionary<String,[]>)` |  |
| `ComputePerturbation(Dictionary<String,[]>)` | Computes the SAM perturbation direction from the current gradient. |
| `ComputeSAMGradient(Dictionary<String,[]>,Dictionary<String,[]>)` | Computes the complete SAM-modified gradient for a training step. |
| `ComputeSMOOPerturbation(Dictionary<String,[]>)` |  |
| `GetStrategyName` |  |
| `UpdateGradientHistory(Dictionary<String,[]>)` | Updates the gradient history (used by FedSpeed/FedLESAM for approximation). |

